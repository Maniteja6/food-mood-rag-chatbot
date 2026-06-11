"""
vector_store/base.py
════════════════════
Abstract base class that defines the contract every vector store
implementation must satisfy.

All concrete stores (ChromaStore, FAISSStore) inherit from VectorStoreBase
and must implement every abstract method.  This lets the rest of the
codebase — pipeline.py, retriever.py, ingest.py — work with any backend
through a single, stable interface.

Interface summary
─────────────────
    # Write
    store.upsert(ids, vectors, documents, metadatas)
    store.delete(ids)
    store.delete_all()

    # Read
    store.query(vector, top_k)        → list[QueryResult]
    store.get(ids)                    → list[dict]
    store.exists(id)                  → bool
    store.count()                     → int

    # Lifecycle
    store.persist()                   → None  (no-op for always-persistent stores)
    store.health()                    → dict

    # Factory
    VectorStoreBase.from_settings()   → VectorStoreBase  (calls correct subclass)
    VectorStoreBase.from_config(cfg)  → VectorStoreBase

Public types
────────────
    QueryResult   TypedDict
        chunk_id   str
        document   str
        metadata   dict
        score      float    cosine similarity 0–1
        rank       int
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QueryResult TypedDict
# ─────────────────────────────────────────────────────────────────────────────

class QueryResult(TypedDict):
    """
    A single result returned by VectorStoreBase.query().

    chunk_id   Unique ID matching the ingestion Chunk.chunk_id.
    document   The full embeddable text string (may be empty for FAISS).
    metadata   Flat dict of scalar metadata fields stored alongside the vector.
    score      Cosine similarity score in [0, 1].  Higher = more similar.
    rank       1-based position in the result list after sorting.
    """
    chunk_id: str
    document: str
    metadata: dict
    score:    float
    rank:     int


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreBase(ABC):
    """
    Abstract base class for MoodBite vector store backends.

    Subclasses must implement all abstract methods.  Concrete implementations
    are ChromaStore (vector_store/chroma_store.py) and FAISSStore
    (vector_store/faiss_store.py).

    Factory usage
    ─────────────
        store = VectorStoreBase.from_settings()           # reads .env
        store = VectorStoreBase.from_config(cfg_dict)     # raw dict

    These return the correct concrete subclass based on
    VECTOR_STORE_PROVIDER in the config.
    """

    # ── Abstract write methods ────────────────────────────────────────────────

    @abstractmethod
    def upsert(
        self,
        ids:       list[str],
        vectors:   list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Insert or update vectors and their associated data.

        If a record with a given id already exists it is overwritten.
        All four lists must be the same length.

        Args:
            ids:       Unique string IDs (e.g. chunk_id from the Chunk object).
            vectors:   Dense float vectors from the embedding model.
            documents: Raw embeddable text strings (one per vector).
            metadatas: Flat scalar dicts stored alongside each vector.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Remove vectors by ID.  Missing IDs are silently ignored.

        Args:
            ids: List of chunk_id strings to remove.
        """

    @abstractmethod
    def delete_all(self) -> None:
        """
        Remove all vectors from this store / collection.
        Used by the ``--force`` flag during ingestion.
        """

    # ── Abstract read methods ─────────────────────────────────────────────────

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k:  int = 10,
    ) -> list[QueryResult]:
        """
        Return the ``top_k`` most similar vectors to ``vector``.

        Results are sorted by cosine similarity score descending.
        Assigns 1-based ranks before returning.

        Args:
            vector: Query vector (same dimension as stored vectors).
            top_k:  Maximum number of results to return.

        Returns:
            List of QueryResult dicts, length ≤ top_k.
        """

    @abstractmethod
    def get(self, ids: list[str]) -> list[dict]:
        """
        Fetch stored records by ID.

        Args:
            ids: List of chunk_id strings.

        Returns:
            List of dicts with keys {chunk_id, document, metadata}.
            Records not found are omitted from the list.
        """

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        """
        Return True if a record with the given chunk_id exists.

        Args:
            chunk_id: The chunk_id string to check.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of vectors currently stored."""

    # ── Abstract lifecycle methods ────────────────────────────────────────────

    @abstractmethod
    def persist(self) -> None:
        """
        Flush any in-memory state to disk.

        ChromaDB's PersistentClient writes on every operation so this is a
        no-op for ChromaStore.  FAISSStore calls faiss.write_index() here.
        """

    @abstractmethod
    def health(self) -> dict:
        """
        Return a dict summarising the store's health and configuration.

        Minimum required keys:
            provider      str   "chroma" | "faiss"
            count         int   number of vectors stored
            is_ready      bool  True if the store has at least 1 vector
            path          str   filesystem path to the DB
        """

    # ── Factory methods (concrete) ────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "VectorStoreBase":
        """
        Build the correct VectorStoreBase subclass from the app Settings.

        Args:
            settings: Settings instance.  Reads .env if None.

        Returns:
            Configured ChromaStore or FAISSStore.
        """
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()

        provider = settings.vector_store_provider.value
        return cls._build(provider, settings=settings)

    @classmethod
    def from_config(cls, cfg: dict) -> "VectorStoreBase":
        """
        Build a VectorStoreBase subclass from a raw config dict.
        Used by ``ingestion/ingest.py`` which builds its own dict.

        Expected keys:
            vector_store_provider   str   "chroma" | "faiss"
            vector_db_path          str
            chroma_collection       str   (chroma only)

        Returns:
            Configured ChromaStore or FAISSStore.
        """
        provider = cfg.get("vector_store_provider", "chroma").lower()
        return cls._build(provider, cfg=cfg)

    @classmethod
    def _build(
        cls,
        provider: str,
        settings: Any    = None,
        cfg:      dict   = None,
    ) -> "VectorStoreBase":
        """Internal factory dispatcher."""
        from vector_store.chroma_store import ChromaStore
        from vector_store.faiss_store  import FAISSStore

        if provider == "chroma":
            if settings:
                return ChromaStore.from_settings(settings)
            return ChromaStore.from_config(cfg or {})

        elif provider == "faiss":
            if settings:
                return FAISSStore.from_settings(settings)
            return FAISSStore.from_config(cfg or {})

        else:
            raise ValueError(
                f"Unknown VECTOR_STORE_PROVIDER '{provider}'. "
                "Choose 'chroma' or 'faiss' in your .env file."
            )

    # ── Shared helpers (concrete, inherited by subclasses) ────────────────────

    def get_all_ids(self) -> list[str]:
        """
        Return all stored chunk IDs.

        Default implementation calls get([]) which concrete classes may
        override with a more efficient bulk-fetch.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_all_ids(). "
            "Override this method for efficient ID listing."
        )

    def missing_ids(self, ids: list[str]) -> list[str]:
        """
        Return the subset of ``ids`` that are NOT in the store.
        Useful for incremental ingestion (skip already-indexed chunks).

        Default: calls exists() per id — subclasses may override for speed.
        """
        return [i for i in ids if not self.exists(i)]

    def __repr__(self) -> str:
        h = self.health()
        return (
            f"{type(self).__name__}("
            f"count={h.get('count', '?')}, "
            f"path={h.get('path', '?')!r})"
        )