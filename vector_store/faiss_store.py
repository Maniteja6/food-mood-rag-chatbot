"""
vector_store/faiss_store.py
═══════════════════════════
FAISS-backed vector store for MoodBite.

FAISS (Facebook AI Similarity Search) is the lightweight alternative to
ChromaDB.  It uses a flat inner-product index (IndexFlatIP) which delivers
exact nearest-neighbour search with no approximation.

Key behaviours
──────────────
- Vectors are L2-normalised before storage so inner-product == cosine similarity.
- Metadata is stored in a parallel JSON file (``metadata.json``) keyed by
  integer position in the FAISS index.
- The index is persisted only when ``persist()`` or ``upsert()`` is called
  (unlike ChromaDB which writes on every operation).
- ``delete()`` is O(n) — FAISS does not support in-place deletion.
  We rebuild the index without the deleted IDs.
- ``get_all_ids()`` reads the metadata array, which is always in sync with
  the index because upsert() and delete() keep them together.

When to prefer FAISS over ChromaDB
────────────────────────────────────
- You need the lightest possible dependency (faiss-cpu is a single .so)
- You are deploying to an environment where chromadb's SQLite dependency
  causes issues
- You don't need metadata filtering (FAISS has no WHERE clause)
- Pure query speed is the priority (FAISS is faster at raw ANN search)

Usage
─────
    from vector_store.faiss_store import FAISSStore

    store = FAISSStore.from_settings()
    store.upsert(ids, vectors, documents, metas)
    results = store.query(query_vector, top_k=5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from vector_store.base import VectorStoreBase, QueryResult

logger = logging.getLogger(__name__)

# Filenames inside the vector_db_path directory
_INDEX_FILE = "index.faiss"
_META_FILE  = "metadata.json"


class FAISSStore(VectorStoreBase):
    """
    FAISS IndexFlatIP vector store with a parallel JSON metadata sidecar.

    In-memory state
    ───────────────
    self._index   faiss.IndexFlatIP  (None until first upsert or load)
    self._meta    list[dict]          parallel metadata array
    self._id_map  dict[str, int]      chunk_id → index position (for fast lookup)

    The _meta list and _id_map are always kept in sync with _index.
    """

    def __init__(
        self,
        db_path: str | Path,
    ) -> None:
        self._db_path   = Path(db_path)
        self._index     = None         # faiss.IndexFlatIP, lazy-loaded
        self._meta:     list[dict] = []
        self._id_map:   dict[str, int] = {}   # chunk_id → int position
        self._dirty     = False               # True if index needs persist()

        self._db_path.mkdir(parents=True, exist_ok=True)
        self._load_if_exists()

        logger.info(
            f"FAISSStore configured: path='{self._db_path}'  "
            f"vectors={len(self._meta):,}"
        )

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "FAISSStore":
        """Build from Settings object (reads .env if None)."""
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        return cls(db_path=settings.vector_db_path)

    @classmethod
    def from_config(cls, cfg: dict) -> "FAISSStore":
        """Build from raw config dict (used by ingestion pipeline)."""
        return cls(db_path=cfg.get("vector_db_path", "./data/vector_db"))

    # ── Internal: lazy load ───────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        """Load existing index and metadata files from disk if they exist."""
        index_path = self._db_path / _INDEX_FILE
        meta_path  = self._db_path / _META_FILE

        if not index_path.exists():
            logger.debug("No existing FAISS index found — starting fresh.")
            return

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS vector store. "
                "Run: pip install faiss-cpu numpy"
            )

        try:
            self._index = faiss.read_index(str(index_path))
        except Exception as exc:                               # noqa: BLE001
            logger.error(f"Failed to load FAISS index from '{index_path}': {exc}")
            self._index = None
            return

        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as fh:
                self._meta = json.load(fh)
        else:
            logger.warning(
                f"FAISS index found at '{index_path}' but no metadata file at "
                f"'{meta_path}'. Metadata will be empty."
            )
            self._meta = [{} for _ in range(self._index.ntotal)]

        # Rebuild the id_map from the metadata
        self._id_map = {
            m.get("chunk_id", str(i)): i
            for i, m in enumerate(self._meta)
        }

        logger.info(
            f"Loaded FAISS index: {self._index.ntotal:,} vectors  "
            f"dim={self._index.d}"
        )

    def _require_faiss(self):
        """Import faiss + numpy; raise a clear error if missing."""
        try:
            import faiss
            import numpy as np
            return faiss, np
        except ImportError:
            raise ImportError(
                "faiss-cpu and numpy are required. "
                "Run: pip install faiss-cpu numpy"
            )

    def _init_index(self, dim: int) -> None:
        """Create a new empty IndexFlatIP for the given dimension."""
        faiss, _ = self._require_faiss()
        self._index = faiss.IndexFlatIP(dim)
        logger.info(f"Created new FAISS IndexFlatIP(dim={dim}).")

    # ── Write methods ─────────────────────────────────────────────────────────

    def upsert(
        self,
        ids:       list[str],
        vectors:   list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Insert or update vectors.

        FAISS does not support in-place update, so for existing IDs we:
            1. Remove the old records (rebuilds index without them)
            2. Append the new records

        New IDs are simply appended.

        All four lists must be the same length.

        Args:
            ids:       Unique string IDs.
            vectors:   Float vectors (will be L2-normalised internally).
            documents: Raw text strings (stored in metadata sidecar).
            metadatas: Flat scalar dicts.
        """
        if not ids:
            return
        if not (len(ids) == len(vectors) == len(documents) == len(metadatas)):
            raise ValueError(
                "upsert() requires all lists to be the same length. "
                f"Got ids={len(ids)}, vectors={len(vectors)}, "
                f"documents={len(documents)}, metadatas={len(metadatas)}."
            )

        faiss, np = self._require_faiss()

        # Separate new IDs from updates
        existing_ids = [i for i in ids if i in self._id_map]
        new_ids      = [i for i in ids if i not in self._id_map]

        # Remove existing IDs first (if any) — triggers index rebuild
        if existing_ids:
            self.delete(existing_ids)

        # Prepare vectors matrix
        mat = np.array(vectors, dtype="float32")
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)

        # Validate / create index
        dim = mat.shape[1]
        if self._index is None:
            self._init_index(dim)
        elif self._index.d != dim:
            raise ValueError(
                f"Vector dimension mismatch: index has dim={self._index.d}, "
                f"but new vectors have dim={dim}."
            )

        # L2-normalise so inner-product == cosine similarity
        faiss.normalize_L2(mat)

        # Add to index
        start_pos = self._index.ntotal
        self._index.add(mat)

        # Update metadata sidecar and id_map
        for i, (chunk_id, doc, meta) in enumerate(
            zip(ids, documents, metadatas)
        ):
            enriched_meta = dict(meta)
            enriched_meta["chunk_id"] = chunk_id
            enriched_meta["_document"] = doc      # store doc text in metadata
            self._meta.append(enriched_meta)
            self._id_map[chunk_id] = start_pos + i

        self._dirty = True
        logger.debug(
            f"Upserted {len(ids)} vectors into FAISS index "
            f"(total: {self._index.ntotal:,})."
        )

    def delete(self, ids: list[str]) -> None:
        """
        Delete vectors by ID.

        FAISS does not support in-place deletion — we rebuild the entire
        index from the remaining vectors.  This is O(n) but acceptable for
        our use case (deletion only happens during --force ingestion).

        Args:
            ids: List of chunk_id strings to remove.
        """
        if not ids or self._index is None or self._index.ntotal == 0:
            return

        faiss, np = self._require_faiss()

        delete_set = set(ids)
        keep_positions = [
            i for i, m in enumerate(self._meta)
            if m.get("chunk_id") not in delete_set
        ]

        if not keep_positions:
            # Delete everything
            self._index.reset()
            self._meta   = []
            self._id_map = {}
            self._dirty  = True
            logger.debug(f"Deleted all {len(ids)} vectors from FAISS index.")
            return

        if len(keep_positions) == len(self._meta):
            logger.debug("delete() called with IDs not in index — no-op.")
            return

        # Reconstruct index from kept vectors
        dim        = self._index.d
        kept_count = len(keep_positions)

        # Fetch all stored vectors
        all_vecs = np.zeros((self._index.ntotal, dim), dtype="float32")
        for pos in range(self._index.ntotal):
            all_vecs[pos] = self._index.reconstruct(pos)

        # Build new index with only kept vectors
        new_index = faiss.IndexFlatIP(dim)
        kept_vecs = all_vecs[keep_positions]
        if len(kept_vecs) > 0:
            new_index.add(kept_vecs)

        # Rebuild metadata and id_map
        new_meta  = [self._meta[i] for i in keep_positions]
        new_id_map = {
            m.get("chunk_id", str(i)): i
            for i, m in enumerate(new_meta)
        }

        self._index  = new_index
        self._meta   = new_meta
        self._id_map = new_id_map
        self._dirty  = True

        removed = len(ids) - (len(self._meta) - kept_count)
        logger.debug(
            f"Deleted {removed} vectors from FAISS index "
            f"(remaining: {len(new_meta):,})."
        )

    def delete_all(self) -> None:
        """Remove all vectors and metadata."""
        if self._index is not None:
            self._index.reset()
        self._meta   = []
        self._id_map = {}
        self._dirty  = True
        # Also remove files from disk
        for fname in (_INDEX_FILE, _META_FILE):
            p = self._db_path / fname
            if p.exists():
                p.unlink()
        logger.info("FAISSStore: deleted all vectors and removed index files.")

    # ── Read methods ──────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float],
        top_k:  int = 10,
    ) -> list[QueryResult]:
        """
        Return the top_k most similar vectors to the query vector.

        Normalises the query vector before searching so the inner product
        equals the cosine similarity.  Clips scores to [0, 1].

        Args:
            vector: Query float vector (raw, will be normalised internally).
            top_k:  Maximum results to return.

        Returns:
            List of QueryResult dicts sorted by score descending.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISSStore.query() called on empty index.")
            return []

        faiss, np = self._require_faiss()

        k   = min(top_k, self._index.ntotal)
        vec = np.array([vector], dtype="float32")
        faiss.normalize_L2(vec)

        scores_arr, idx_arr = self._index.search(vec, k)
        raw_scores  = scores_arr[0].tolist()
        raw_indices = idx_arr[0].tolist()

        results: list[QueryResult] = []
        for rank, (idx, score) in enumerate(
            zip(raw_indices, raw_scores), start=1
        ):
            if idx < 0 or idx >= len(self._meta):
                continue    # FAISS pads with -1 for small indexes

            meta     = dict(self._meta[idx])
            doc_text = meta.pop("_document", "")    # retrieve stored doc text
            chunk_id = meta.get("chunk_id", str(idx))

            results.append(QueryResult(
                chunk_id=chunk_id,
                document=doc_text,
                metadata=meta,
                score=round(max(0.0, min(1.0, float(score))), 4),
                rank=rank,
            ))

        return results

    def get(self, ids: list[str]) -> list[dict]:
        """
        Fetch stored records by ID from the metadata sidecar.

        Args:
            ids: List of chunk_id strings.

        Returns:
            List of dicts with keys {chunk_id, document, metadata}.
        """
        results = []
        for chunk_id in ids:
            pos = self._id_map.get(chunk_id)
            if pos is None or pos >= len(self._meta):
                continue
            meta     = dict(self._meta[pos])
            doc_text = meta.pop("_document", "")
            results.append({
                "chunk_id": chunk_id,
                "document": doc_text,
                "metadata": meta,
            })
        return results

    def exists(self, chunk_id: str) -> bool:
        """Return True if the chunk_id is present in the index."""
        return chunk_id in self._id_map

    def count(self) -> int:
        """Return the total number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def get_all_ids(self) -> list[str]:
        """Return all stored chunk IDs from the metadata sidecar."""
        return [m.get("chunk_id", "") for m in self._meta]

    def missing_ids(self, ids: list[str]) -> list[str]:
        """Return the subset of ids not yet in the index."""
        return [i for i in ids if i not in self._id_map]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def persist(self) -> None:
        """
        Flush the FAISS index and metadata sidecar to disk.

        Call this after every batch of upsert() calls during ingestion.
        The pipeline's ingest.py calls persist() automatically at the end
        of the embed-and-upsert loop.
        """
        if self._index is None:
            logger.debug("persist() called on empty FAISSStore — nothing to write.")
            return

        faiss, _ = self._require_faiss()
        index_path = self._db_path / _INDEX_FILE
        meta_path  = self._db_path / _META_FILE

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, ensure_ascii=False)

        self._dirty = False
        size_mb = index_path.stat().st_size / 1024 / 1024
        logger.info(
            f"FAISSStore persisted: {self._index.ntotal:,} vectors → "
            f"'{index_path}' ({size_mb:.1f} MB)"
        )

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return health and configuration summary."""
        count    = self.count()
        is_ready = count > 0
        dim      = self._index.d if self._index else None

        return {
            "provider":   "faiss",
            "path":       str(self._db_path),
            "count":      count,
            "dimension":  dim,
            "is_ready":   is_ready,
            "dirty":      self._dirty,
            "index_file": str(self._db_path / _INDEX_FILE),
            "meta_file":  str(self._db_path / _META_FILE),
        }