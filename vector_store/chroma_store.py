"""
vector_store/chroma_store.py
════════════════════════════
ChromaDB-backed vector store for MoodBite.

ChromaDB is the default vector store — it persists automatically to disk,
supports cosine similarity natively, allows metadata filtering, and has a
Python-native API with no server process required.

Key behaviours
──────────────
- Uses PersistentClient so the collection survives process restarts.
- Collection is created with ``hnsw:space: "cosine"`` so all distances are
  cosine distances in [0, 2]; we convert to similarity with 1 - distance.
- upsert() is idempotent — existing IDs are silently overwritten.
- delete_all() drops and re-creates the collection rather than iterating IDs,
  which is O(1) regardless of collection size.
- get_all_ids() fetches IDs in pages of 1 000 to avoid memory spikes on
  large collections.

Usage
─────
    from vector_store.chroma_store import ChromaStore

    store = ChromaStore.from_settings()         # reads .env
    store.upsert(ids, vectors, documents, metas)
    results = store.query(query_vector, top_k=5)
    print(store.count())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from vector_store.base import VectorStoreBase, QueryResult

logger = logging.getLogger(__name__)

# Page size used when listing all IDs
_ID_PAGE_SIZE = 1_000


class ChromaStore(VectorStoreBase):
    """
    ChromaDB persistent vector store.

    Parameters
    ──────────
    db_path
        Directory where ChromaDB stores its SQLite + HNSW index files.
        Created automatically if it does not exist.
    collection_name
        Name of the ChromaDB collection.  One collection = one topic space.
    """

    def __init__(
        self,
        db_path:         str | Path,
        collection_name: str = "food_mood_collection",
    ) -> None:
        self._db_path         = Path(db_path)
        self._collection_name = collection_name
        self._client          = None   # lazy-init
        self._collection      = None   # lazy-init

        self._db_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"ChromaStore configured: path='{self._db_path}'  "
            f"collection='{self._collection_name}'"
        )

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "ChromaStore":
        """Build from Settings object (reads .env if None)."""
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        return cls(
            db_path=settings.vector_db_path,
            collection_name=settings.chroma_collection_name,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "ChromaStore":
        """Build from raw config dict (used by ingestion pipeline)."""
        return cls(
            db_path=cfg.get("vector_db_path", "./data/vector_db"),
            collection_name=cfg.get("chroma_collection", "food_mood_collection"),
        )

    # ── Internal: lazy client + collection ───────────────────────────────────

    def _get_collection(self):
        """Lazy-init the ChromaDB client and collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

        self._client = chromadb.PersistentClient(path=str(self._db_path))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(
            f"ChromaDB collection '{self._collection_name}' opened "
            f"({count:,} vectors)."
        )
        if count == 0:
            logger.warning(
                "Collection is empty. Run 'make ingest' to populate the vector database."
            )
        return self._collection

    # ── Write methods ─────────────────────────────────────────────────────────

    def upsert(
        self,
        ids:       list[str],
        vectors:   list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Insert or update vectors in the collection.

        All four lists must be the same length.  Existing IDs are silently
        overwritten (upsert semantics).

        Args:
            ids:       Unique string IDs (chunk_id values).
            vectors:   Float vectors from the embedding model.
            documents: Embeddable text strings (one per vector).
            metadatas: Flat scalar dicts (no nested objects or lists).
        """
        if not ids:
            return

        if not (len(ids) == len(vectors) == len(documents) == len(metadatas)):
            raise ValueError(
                f"upsert() requires all lists to be the same length. "
                f"Got ids={len(ids)}, vectors={len(vectors)}, "
                f"documents={len(documents)}, metadatas={len(metadatas)}."
            )

        col = self._get_collection()
        col.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug(f"Upserted {len(ids)} records into '{self._collection_name}'.")

    def delete(self, ids: list[str]) -> None:
        """
        Delete records by ID.  Missing IDs are silently ignored.

        Args:
            ids: List of chunk_id strings to remove.
        """
        if not ids:
            return
        col = self._get_collection()
        col.delete(ids=ids)
        logger.debug(f"Deleted {len(ids)} records from '{self._collection_name}'.")

    def delete_all(self) -> None:
        """
        Drop the entire collection and recreate it empty.

        This is O(1) regardless of collection size, much faster than
        iterating and deleting individual records.
        """
        if self._client is None:
            # Force client init before deleting
            self._get_collection()

        try:
            self._client.delete_collection(self._collection_name)
            logger.info(f"Dropped ChromaDB collection '{self._collection_name}'.")
        except Exception as exc:                                # noqa: BLE001
            logger.warning(f"Could not drop collection: {exc}")

        # Recreate empty
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Recreated empty collection '{self._collection_name}'.")

    # ── Read methods ──────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float],
        top_k:  int = 10,
    ) -> list[QueryResult]:
        """
        Return the top_k most similar vectors to the query vector.

        Converts ChromaDB cosine distances to similarity scores:
            similarity = 1 - distance     (distance ∈ [0, 2])
        Clips to [0, 1] to handle floating-point noise.

        Args:
            vector: Query float vector.
            top_k:  Maximum results to return.

        Returns:
            List of QueryResult dicts sorted by score descending.
        """
        col = self._get_collection()
        n   = min(top_k, col.count())
        if n == 0:
            return []

        results = col.query(
            query_embeddings=[vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        ids       = results.get("ids",       [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: list[QueryResult] = []
        for rank, (chunk_id, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            output.append(QueryResult(
                chunk_id=chunk_id,
                document=doc or "",
                metadata=meta or {},
                score=round(score, 4),
                rank=rank,
            ))

        return output

    def get(self, ids: list[str]) -> list[dict]:
        """
        Fetch stored records by ID.

        Args:
            ids: List of chunk_id strings.

        Returns:
            List of dicts with keys {chunk_id, document, metadata}.
        """
        if not ids:
            return []

        col     = self._get_collection()
        results = col.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        out_ids       = results.get("ids",       [])
        out_documents = results.get("documents", [])
        out_metadatas = results.get("metadatas", [])

        return [
            {
                "chunk_id": cid,
                "document": doc or "",
                "metadata": meta or {},
            }
            for cid, doc, meta in zip(out_ids, out_documents, out_metadatas)
        ]

    def exists(self, chunk_id: str) -> bool:
        """Return True if a record with the given chunk_id exists."""
        col     = self._get_collection()
        results = col.get(ids=[chunk_id], include=[])
        return bool(results.get("ids"))

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self._get_collection().count()

    def get_all_ids(self) -> list[str]:
        """
        Return all stored chunk IDs.

        Fetches in pages of _ID_PAGE_SIZE to avoid loading the entire
        collection into memory at once on large databases.
        """
        col    = self._get_collection()
        total  = col.count()
        if total == 0:
            return []

        all_ids: list[str] = []
        offset = 0
        while offset < total:
            page = col.get(
                include=[],
                limit=_ID_PAGE_SIZE,
                offset=offset,
            )
            page_ids = page.get("ids", [])
            if not page_ids:
                break
            all_ids.extend(page_ids)
            offset += len(page_ids)

        return all_ids

    def missing_ids(self, ids: list[str]) -> list[str]:
        """
        Return the subset of ``ids`` that are NOT in the collection.
        More efficient than the base class implementation for large batches
        because it fetches all existing IDs once rather than calling exists()
        per ID.
        """
        if not ids:
            return []
        existing = set(self.get_all_ids())
        return [i for i in ids if i not in existing]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def persist(self) -> None:
        """
        No-op for ChromaDB PersistentClient.
        ChromaDB writes every operation immediately to the SQLite + HNSW files.
        """
        logger.debug("ChromaStore.persist() called — no-op (always persistent).")

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return health and configuration summary."""
        try:
            count    = self.count()
            is_ready = count > 0
        except Exception:                                       # noqa: BLE001
            count    = 0
            is_ready = False

        return {
            "provider":         "chroma",
            "collection_name":  self._collection_name,
            "path":             str(self._db_path),
            "count":            count,
            "is_ready":         is_ready,
        }