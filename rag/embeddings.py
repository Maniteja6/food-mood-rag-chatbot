"""
rag/embeddings.py
═════════════════
Embedding model abstraction for MoodBite.

Provides a single ``EmbeddingModel`` class that wraps both supported
embedding backends — OpenAI and HuggingFace sentence-transformers — behind
one consistent interface.  Both the RAG pipeline (query-time) and the
ingestion pipeline (index-time) use this class so the same model and
normalisation logic is guaranteed in both directions.

Supported providers
───────────────────
openai
    Calls the OpenAI Embeddings API (``text-embedding-3-small`` by default).
    Produces 1536-dim vectors.  Requires ``OPENAI_API_KEY`` in ``.env``.
    Batches requests automatically; retries on rate-limit errors.

huggingface
    Runs a ``sentence-transformers`` model locally — no API key needed.
    Default: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim).
    Works offline and on Streamlit Community Cloud without secrets.
    First call downloads the model weights (~90 MB); subsequent calls are
    instant because the model stays in memory.

Public API
──────────
    from rag.embeddings import EmbeddingModel

    em = EmbeddingModel.from_settings()        # build from .env config
    em = EmbeddingModel.from_config(cfg_dict)  # build from raw dict

    vector  = em.embed_query("something warm and comforting")
    vectors = em.embed_documents(["doc1 text", "doc2 text", ...])
    info    = em.info()                        # provider / model / dim

Design decisions
────────────────
- ``embed_query`` and ``embed_documents`` are separate methods (not one
  ``embed``) because some providers (e.g. Cohere, future) use different
  instruction prefixes for queries vs documents.  For OpenAI and HuggingFace
  they call the same underlying function, but the separation keeps the
  interface future-proof.

- Vectors are NOT L2-normalised here.  The ingestion pipeline normalises
  before writing to FAISS (``faiss.normalize_L2``).  The retriever
  normalises before querying FAISS.  ChromaDB handles normalisation
  internally when the collection uses cosine distance.  Keeping raw vectors
  from the embedding model means callers can choose their own norm strategy.

- Thread safety: the HuggingFace SentenceTransformer is loaded once and
  reused across calls.  It is safe to share across Streamlit reruns via
  ``@st.cache_resource`` because the model object is read-only at inference
  time.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingModel
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Provider-agnostic embedding wrapper.

    Do not instantiate directly — use the factory class methods:

        EmbeddingModel.from_settings()       reads from .env via get_settings()
        EmbeddingModel.from_config(cfg)      reads from a raw config dict
        EmbeddingModel(provider, **kwargs)   direct construction
    """

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        provider:   str,
        model_name: str,
        api_key:    str  = "",
        batch_size: int  = 100,
    ) -> None:
        """
        Args:
            provider:   "openai" or "huggingface"
            model_name: e.g. "text-embedding-3-small" or
                        "sentence-transformers/all-MiniLM-L6-v2"
            api_key:    OpenAI API key (ignored for HuggingFace)
            batch_size: Max texts per API call (OpenAI) or encode() call (HF)
        """
        self._provider   = provider.lower().strip()
        self._model_name = model_name
        self._api_key    = api_key
        self._batch_size = max(1, batch_size)

        # Lazy-loaded backends
        self._openai_client: Any = None
        self._hf_model:      Any = None
        self._dimension:     int | None = None

        self._validate_provider()
        logger.info(
            f"EmbeddingModel({self._provider}/{self._model_name}  "
            f"batch_size={self._batch_size})"
        )

    def _validate_provider(self) -> None:
        if self._provider not in ("openai", "huggingface"):
            raise ValueError(
                f"Unknown embedding provider '{self._provider}'. "
                "Choose 'openai' or 'huggingface'."
            )

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "EmbeddingModel":
        """
        Build an EmbeddingModel from the application Settings singleton.

        Args:
            settings: Optional Settings instance.  If None, calls
                      ``config.settings.get_settings()``.

        Returns:
            Configured EmbeddingModel.
        """
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()

        provider = settings.embedding_provider.value
        model    = (
            settings.hf_embedding_model
            if provider == "huggingface"
            else settings.embedding_model
        )
        return cls(
            provider=provider,
            model_name=model,
            api_key=settings.openai_api_key,
            batch_size=settings.embed_batch_size,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "EmbeddingModel":
        """
        Build an EmbeddingModel from a raw config dict.
        Used by ``ingestion/ingest.py`` which builds its own config dict
        from env vars rather than the Settings object.

        Expected keys (all optional, fall back to safe defaults):
            embedding_provider   str   "openai" | "huggingface"
            embedding_model      str   OpenAI model name
            hf_embedding_model   str   HuggingFace model name
            openai_api_key       str
            batch_size           int

        Returns:
            Configured EmbeddingModel.
        """
        provider = cfg.get("embedding_provider", "huggingface").lower()
        model    = (
            cfg.get("hf_embedding_model",
                    "sentence-transformers/all-MiniLM-L6-v2")
            if provider == "huggingface"
            else cfg.get("embedding_model", "text-embedding-3-small")
        )
        return cls(
            provider=provider,
            model_name=model,
            api_key=cfg.get("openai_api_key", ""),
            batch_size=int(cfg.get("batch_size", 100)),
        )

    # ── Public embedding methods ──────────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            text: The user's (optionally mood-expanded) query.

        Returns:
            1-D list of floats (the embedding vector).

        Raises:
            RuntimeError: if the API key is missing for OpenAI.
            ImportError:  if the required package is not installed.
        """
        if not text or not text.strip():
            raise ValueError("embed_query() received an empty string.")
        vectors = self._embed_batch([text.strip()])
        return vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document strings in batches.

        Args:
            texts: List of document strings (chunk documents from chunker.py).

        Returns:
            List of float vectors, same length and order as ``texts``.

        Raises:
            ValueError: if ``texts`` is empty.
        """
        if not texts:
            raise ValueError("embed_documents() received an empty list.")

        # Filter out blanks and remember their positions for reinsertion
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts   = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            raise ValueError("All texts in embed_documents() are blank.")

        # Batch embed
        vectors: list[list[float]] = []
        total     = len(non_empty_texts)
        inserted  = 0
        t_start   = time.perf_counter()

        for start in range(0, total, self._batch_size):
            batch = non_empty_texts[start : start + self._batch_size]
            batch_vectors = self._embed_batch(batch)
            vectors.extend(batch_vectors)
            inserted += len(batch)

            if total > self._batch_size:
                elapsed = time.perf_counter() - t_start
                pct     = inserted / total * 100
                logger.debug(
                    f"  embed_documents [{inserted:>6,}/{total:,}]  "
                    f"{pct:5.1f}%  "
                    f"{elapsed:.1f}s elapsed"
                )

        # Re-insert zero vectors for any blank inputs (keeps index alignment)
        if len(non_empty_indices) < len(texts):
            dim          = len(vectors[0]) if vectors else self.dimension
            full_vectors = [[0.0] * dim for _ in range(len(texts))]
            
            for idx, vec in zip(non_empty_indices, vectors):
                full_vectors[idx] = vec
            return full_vectors

        return vectors

    # ── Dimension probe ───────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension (number of floats per vector).

        Computed lazily on first call by embedding a single probe string.
        Cached for subsequent calls.
        """
        if self._dimension is None:
            probe = self._embed_batch(["dimension probe"])[0]
            self._dimension = len(probe)
            logger.info(
                f"Embedding dimension: {self._dimension} "
                f"({self._provider}/{self._model_name})"
            )
        return self._dimension

    # ── Model info ────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """
        Return a dict summarising the embedding configuration.
        Used by ``RAGPipeline.health_check()`` and debug UI panels.
        """
        return {
            "provider":   self._provider,
            "model_name": self._model_name,
            "batch_size": self._batch_size,
            "dimension":  self._dimension,      # None if not yet probed
        }

    def __repr__(self) -> str:
        dim = f"dim={self._dimension}" if self._dimension else "dim=?"
        return (
            f"EmbeddingModel(provider={self._provider!r}, "
            f"model={self._model_name!r}, {dim})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: unified batch embed dispatcher
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a single batch of texts using the configured provider.

        This is the single internal method that all public methods call.
        It dispatches to the provider-specific implementation and handles
        retries for transient API errors.

        Args:
            texts: Non-empty list of strings; length ≤ self._batch_size.

        Returns:
            List of float vectors, same length as texts.
        """
        if self._provider == "openai":
            return self._embed_openai(texts)
        elif self._provider == "huggingface":
            return self._embed_huggingface(texts)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    # ─────────────────────────────────────────────────────────────────────────
    # Provider: OpenAI
    # ─────────────────────────────────────────────────────────────────────────

    def _get_openai_client(self):
        """Lazy-init and cache the OpenAI client."""
        if self._openai_client is not None:
            return self._openai_client

        try:
            from openai import OpenAI, APIError, RateLimitError
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Run: pip install openai"
            )

        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file, or switch to "
                "EMBEDDING_PROVIDER=huggingface for local (free) embeddings."
            )

        self._openai_client = OpenAI(api_key=self._api_key)
        logger.debug("OpenAI client initialised.")
        return self._openai_client

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """
        Call the OpenAI Embeddings API with automatic retry on rate limits.

        Retry policy:
            Up to 3 attempts with exponential back-off (2s, 4s, 8s).
            Only retries on RateLimitError and transient network errors.
            Raises immediately on AuthenticationError, InvalidRequestError.
        """
        client     = self._get_openai_client()
        max_tries  = 3

        for attempt in range(1, max_tries + 1):
            try:
                response = client.embeddings.create(
                    input=texts,
                    model=self._model_name,
                )
                # Results are returned in the same order as input
                return [item.embedding for item in response.data]

            except Exception as exc:                            # noqa: BLE001
                exc_name = type(exc).__name__

                # Non-retryable errors — raise immediately
                if any(k in exc_name for k in ("Auth", "Invalid", "NotFound")):
                    raise

                if attempt == max_tries:
                    raise

                wait = 2 ** attempt
                logger.warning(
                    f"OpenAI embed error [{exc_name}] "
                    f"(attempt {attempt}/{max_tries}): {exc}. "
                    f"Retrying in {wait}s …"
                )
                time.sleep(wait)

        # Unreachable — loop always raises or returns
        raise RuntimeError("OpenAI embedding failed after all retries.")

    # ─────────────────────────────────────────────────────────────────────────
    # Provider: HuggingFace sentence-transformers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_hf_model(self):
        """Lazy-load and cache the SentenceTransformer model."""
        if self._hf_model is not None:
            return self._hf_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for HuggingFace embeddings. "
                "Run: pip install sentence-transformers"
            )

        logger.info(
            f"Loading HuggingFace model '{self._model_name}' "
            "(this may take a moment on first load) …"
        )
        t0 = time.perf_counter()
        self._hf_model = SentenceTransformer(self._model_name)
        logger.info(
            f"HuggingFace model loaded in {time.perf_counter() - t0:.1f}s  "
            f"(dim={self._hf_model.get_sentence_embedding_dimension()})"
        )
        return self._hf_model

    def _embed_huggingface(self, texts: list[str]) -> list[list[float]]:
        """
        Encode texts using sentence-transformers locally.

        encode() returns a numpy array; we convert to Python lists so the
        output type is identical to the OpenAI provider.
        """
        model       = self._get_hf_model()
        embeddings  = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # keep raw vectors; caller normalises
        )
        return [vec.tolist() for vec in embeddings]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text: str, settings=None) -> list[float]:
    """
    One-shot helper: embed a single query using settings from .env.

    Useful for ad-hoc testing and scripts:

        from rag.embeddings import embed_query
        vec = embed_query("spicy Thai food for a happy mood")

    Args:
        text:     The query string to embed.
        settings: Optional Settings instance (reads .env if None).

    Returns:
        Float vector.
    """
    return EmbeddingModel.from_settings(settings).embed_query(text)


def embed_documents(texts: list[str], settings=None) -> list[list[float]]:
    """
    One-shot helper: embed a list of documents using settings from .env.

    Args:
        texts:    List of document strings.
        settings: Optional Settings instance (reads .env if None).

    Returns:
        List of float vectors.
    """
    return EmbeddingModel.from_settings(settings).embed_documents(texts)