"""
rag/retriever.py
════════════════
Standalone retriever for the MoodBite RAG pipeline.

The Retriever class owns everything related to finding food documents in the
vector store — query execution, result normalisation, post-retrieval filtering,
score thresholding, and mood-aware re-ranking.

pipeline.py delegates all vector-search work here by constructing a Retriever
at initialise() time and calling retriever.retrieve() inside query().

Design principles
─────────────────
- Single responsibility — only retrieval lives here; no LLM, no prompts.
- Provider-agnostic public API — one retrieve() method works for both
  ChromaDB and FAISS; the caller never needs to know which is in use.
- Defensive — every path has a graceful fallback so the pipeline never
  crashes due to empty results, empty indexes, or filter over-restriction.
- Re-rankable — results can optionally be re-scored by a mood-cuisine
  affinity bonus, nudging mood-aligned dishes to the top without
  discarding anything.

Public API
──────────
    Retriever(vector_store, config)
    retriever.retrieve(query_vector, mood, filters)  → list[RetrievedDoc]
    retriever.health()                                → dict

    RetrievedDoc  TypedDict:
        chunk_id   str
        document   str
        metadata   dict
        score      float    cosine similarity 0–1
        rank       int      1-based position after final sorting
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

class RetrievedDoc(TypedDict):
    chunk_id:  str
    document:  str
    metadata:  dict
    score:     float
    rank:      int

Filters = dict[str, list[str]]   # {"dietary": [...], "cuisine": [...]}


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Vector-store retriever with mood-aware filtering and re-ranking.

    Parameters
    ──────────
    vector_store
        Either a ChromaDB Collection object (when provider == "chroma")
        or a dict {"index": faiss.Index, "meta": list[dict], "path": Path}
        (when provider == "faiss").

    config
        The application Settings object. The retriever reads:
            cfg.vector_store_provider    "chroma" | "faiss"
            cfg.retriever_top_k          int   how many results to return
            cfg.retriever_score_threshold float  minimum cosine similarity
    """

    def __init__(self, vector_store: Any, config: Any) -> None:
        self._store  = vector_store
        self._cfg    = config
        self._provider = config.vector_store_provider.value   # "chroma" | "faiss"
        logger.info(
            f"Retriever ready — provider={self._provider}  "
            f"top_k={config.retriever_top_k}  "
            f"threshold={config.retriever_score_threshold}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_vector: list[float],
        mood:         Optional[str]  = None,
        filters:      Optional[Filters] = None,
    ) -> list[RetrievedDoc]:
        """
        Retrieve and rank the most relevant food documents for a query vector.

        Pipeline
        ────────
        1. Raw search       — fetch fetch_k candidates from the vector store
        2. Normalise        — convert provider-specific results to RetrievedDoc
        3. Post-filter      — apply dietary and cuisine hard filters
        4. Score threshold  — drop results below minimum similarity
        5. Mood re-rank     — apply a small bonus to mood-aligned cuisines
        6. Fallback         — if everything was filtered, use top raw results
        7. Trim             — return at most top_k results

        Args:
            query_vector:  Dense float vector from the embedding model.
            mood:          Active mood key e.g. "cozy". Used for re-ranking.
            filters:       {"dietary": [...], "cuisine": [...]}
                           Empty lists mean "no filter applied".

        Returns:
            List of RetrievedDoc dicts, sorted by final score descending,
            length ≤ cfg.retriever_top_k.
        """
        filters = filters or {}
        top_k   = self._cfg.retriever_top_k
        fetch_k = max(top_k * 4, 20)   # over-fetch to allow filter headroom

        # ── Step 1+2: raw search + normalise ─────────────────────────────────
        if self._provider == "chroma":
            raw = self._search_chroma(query_vector, fetch_k)
        elif self._provider == "faiss":
            raw = self._search_faiss(query_vector, fetch_k)
        else:
            raise ValueError(
                f"Unknown vector_store_provider '{self._provider}'. "
                "Set VECTOR_STORE_PROVIDER=chroma or faiss in your .env."
            )

        if not raw:
            logger.warning("Vector store returned 0 results.")
            return []

        # ── Step 3: post-retrieval hard filters ───────────────────────────────
        filtered = self._apply_hard_filters(raw, filters)

        # ── Step 4: score threshold ───────────────────────────────────────────
        threshold = self._cfg.retriever_score_threshold
        above_threshold = [d for d in filtered if d["score"] >= threshold]

        # ── Step 5: mood re-ranking ───────────────────────────────────────────
        if mood:
            above_threshold = self._mood_rerank(above_threshold, mood)

        # ── Step 6: fallback if all results were filtered out ─────────────────
        final = above_threshold
        if not final:
            logger.debug(
                f"All {len(raw)} results filtered/thresholded — "
                "returning top raw results as fallback."
            )
            # Re-rank raw results by mood even in fallback mode
            fallback = raw[:top_k]
            if mood:
                fallback = self._mood_rerank(fallback, mood)
            final = fallback

        # ── Step 7: trim to top_k and assign final ranks ──────────────────────
        final = final[:top_k]
        for i, doc in enumerate(final, start=1):
            doc["rank"] = i

        logger.debug(
            f"retrieve() → {len(final)} docs  "
            f"(raw={len(raw)} filtered={len(filtered)} "
            f"above_threshold={len(above_threshold)})"
        )
        return final

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1+2 — Raw vector search per provider
    # ─────────────────────────────────────────────────────────────────────────

    def _search_chroma(
        self,
        query_vector: list[float],
        fetch_k:      int,
    ) -> list[RetrievedDoc]:
        """
        Query ChromaDB and return normalised results.

        ChromaDB returns cosine *distances* (0 = identical, 2 = opposite).
        We convert to similarity: score = 1 - distance, clipped to [0, 1].
        """
        try:
            results = self._store.query(
                query_embeddings=[query_vector],
                n_results=min(fetch_k, self._store.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:                                # noqa: BLE001
            logger.error(f"ChromaDB query failed: {exc}")
            return []

        ids       = results.get("ids",       [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        docs: list[RetrievedDoc] = []
        for i, (chunk_id, doc_text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            docs.append(RetrievedDoc(
                chunk_id=chunk_id,
                document=doc_text or "",
                metadata=meta or {},
                score=round(score, 4),
                rank=i + 1,
            ))

        return docs

    def _search_faiss(
        self,
        query_vector: list[float],
        fetch_k:      int,
    ) -> list[RetrievedDoc]:
        """
        Query a FAISS IndexFlatIP and return normalised results.

        The ingestion pipeline L2-normalises vectors before adding them, so
        inner-product search is equivalent to cosine similarity. Scores are
        already in [-1, 1]; we clip to [0, 1].
        """
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS vector store. "
                "Run: pip install faiss-cpu numpy"
            )

        index = self._store.get("index")
        meta  = self._store.get("meta", [])

        if index is None or index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return []

        # L2-normalise the query so inner product == cosine similarity
        vec = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(vec)

        k           = min(fetch_k, index.ntotal)
        scores_arr, idx_arr = index.search(vec, k)
        raw_scores  = scores_arr[0].tolist()
        raw_indices = idx_arr[0].tolist()

        docs: list[RetrievedDoc] = []
        for rank, (idx, score) in enumerate(
            zip(raw_indices, raw_scores), start=1
        ):
            if idx < 0 or idx >= len(meta):
                continue                    # FAISS pads with -1 for small indexes
            m = meta[idx]
            docs.append(RetrievedDoc(
                chunk_id=m.get("chunk_id", str(idx)),
                document="",               # FAISS doesn't store raw text
                metadata=m,
                score=round(max(0.0, min(1.0, float(score))), 4),
                rank=rank,
            ))

        return docs

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Hard metadata filters
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_hard_filters(
        self,
        docs:    list[RetrievedDoc],
        filters: Filters,
    ) -> list[RetrievedDoc]:
        """
        Apply user-specified dietary and cuisine hard filters.

        Rules
        ─────
        dietary (OR logic)
            A doc passes if AT LEAST ONE of the user's selected dietary tags
            appears (case-insensitive substring match) in its `dietary_tags`
            metadata field. OR logic keeps results permissive — "Vegetarian"
            still matches "Vegetarian, Gluten-Free".

        cuisine (OR logic, exact match)
            A doc passes if its `cuisine` field exactly matches one of the
            user's selected cuisines (case-insensitive).

        If a filter list is empty, that filter is skipped entirely (no-op).
        """
        dietary_filter = [d.lower().strip() for d in filters.get("dietary", []) if d.strip()]
        cuisine_filter = [c.lower().strip() for c in filters.get("cuisine", []) if c.strip()]

        if not dietary_filter and not cuisine_filter:
            return docs   # fast-path: no filters active

        passed: list[RetrievedDoc] = []
        for doc in docs:
            meta = doc["metadata"]

            # ── Dietary check ─────────────────────────────────────────────────
            if dietary_filter:
                doc_dietary = meta.get("dietary_tags", "").lower()
                if not any(tag in doc_dietary for tag in dietary_filter):
                    continue

            # ── Cuisine check ─────────────────────────────────────────────────
            if cuisine_filter:
                doc_cuisine = meta.get("cuisine", "").lower().strip()
                if doc_cuisine not in cuisine_filter:
                    continue

            passed.append(doc)

        skipped = len(docs) - len(passed)
        if skipped:
            logger.debug(
                f"Hard filters removed {skipped}/{len(docs)} docs "
                f"(dietary={dietary_filter}, cuisine={cuisine_filter})."
            )
        return passed

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Mood-aware re-ranking
    # ─────────────────────────────────────────────────────────────────────────

    def _mood_rerank(
        self,
        docs: list[RetrievedDoc],
        mood: str,
    ) -> list[RetrievedDoc]:
        """
        Apply a small mood-alignment bonus to re-rank results.

        Scoring formula
        ───────────────
        final_score = base_score + (mood_bonus × bonus_weight)

        Where mood_bonus is:
            +0.08   cuisine is in mood's top_cuisines list
            +0.05   doc's moods metadata field contains the active mood
            +0.04   doc's occasion matches one of mood's preferred occasions
            +0.03   doc's flavour_profile matches one of mood's flavours
            +0.02   doc's texture matches one of mood's textures
            +0.02   doc's meal_type matches one of mood's meal_types
            -0.05   doc's spice_level is "Very Spicy" and mood prefers mild

        The bonuses are intentionally small (< 0.25 total) so they can only
        swap closely-scored neighbours — they never surface a low-similarity
        result over a clearly better match.

        The final list is re-sorted by adjusted score descending.
        """
        from config.moods import get_mood
        mood_cfg = get_mood(mood)

        if mood_cfg is None:
            return docs    # unknown mood — skip re-ranking

        top_cuisines   = {c.lower() for c in mood_cfg.top_cuisines}
        pref_occasions = {o.lower() for o in mood_cfg.occasions}
        pref_flavours  = {f.lower() for f in mood_cfg.flavours}
        pref_textures  = {t.lower() for t in mood_cfg.textures}
        pref_meals     = {m.lower() for m in mood_cfg.meal_types}
        prefers_mild   = "mild" in mood_cfg.spice_preference.lower()

        scored: list[tuple[float, RetrievedDoc]] = []
        for doc in docs:
            meta     = doc["metadata"]
            bonus    = 0.0

            # Cuisine alignment
            if meta.get("cuisine", "").lower() in top_cuisines:
                bonus += 0.08

            # Mood tag match (chunk's moods field contains active mood)
            doc_moods = meta.get("moods", "").lower()
            if mood.lower() in doc_moods:
                bonus += 0.05

            # Occasion match
            doc_occasion = meta.get("occasion", "").lower()
            if any(occ in doc_occasion for occ in pref_occasions):
                bonus += 0.04

            # Flavour profile match
            doc_flavour = meta.get("flavour_profile", "").lower()
            if doc_flavour in pref_flavours:
                bonus += 0.03

            # Texture match
            doc_texture = meta.get("texture", "").lower()
            if doc_texture in pref_textures:
                bonus += 0.02

            # Meal type match
            doc_meal = meta.get("meal_type", "").lower()
            if doc_meal in pref_meals:
                bonus += 0.02

            # Spice penalty for mild-preference moods
            doc_spice = meta.get("spice_level", "").lower()
            if prefers_mild and doc_spice == "very spicy":
                bonus -= 0.05

            final = min(1.0, doc["score"] + bonus)

            # Store adjusted score and bonus detail for debug
            reranked = RetrievedDoc(
                chunk_id=doc["chunk_id"],
                document=doc["document"],
                metadata=doc["metadata"],
                score=round(final, 4),
                rank=doc["rank"],
            )
            scored.append((final, reranked))

        # Sort by adjusted score descending, stable sort preserves original
        # order for equal scores
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """
        Return a health/status dict for this retriever instance.
        Used by RAGPipeline.health_check() and debug UI panels.
        """
        vector_count = 0
        try:
            if self._provider == "chroma":
                vector_count = self._store.count()
            elif self._provider == "faiss":
                idx = self._store.get("index")
                if idx is not None:
                    vector_count = idx.ntotal
        except Exception:                                       # noqa: BLE001
            pass

        return {
            "provider":        self._provider,
            "vector_count":    vector_count,
            "top_k":           self._cfg.retriever_top_k,
            "score_threshold": self._cfg.retriever_score_threshold,
            "is_ready":        vector_count > 0,
        }

    def __repr__(self) -> str:
        h = self.health()
        return (
            f"Retriever(provider={h['provider']!r}, "
            f"vectors={h['vector_count']:,}, "
            f"top_k={h['top_k']}, "
            f"threshold={h['score_threshold']})"
        )