"""
rag/pipeline.py
═══════════════
The RAGPipeline class — central orchestrator for the MoodBite chatbot.

Every user message flows through pipeline.query() which:
    1.  Expands the query with mood-specific semantic descriptors
    2.  Embeds the expanded query into a dense vector
    3.  Retrieves the top-K most similar food documents from the vector store
    4.  Optionally filters results by dietary tags and cuisine preferences
    5.  Builds a structured system + user prompt with retrieved context
    6.  Calls the configured LLM (OpenAI / Anthropic / Google)
    7.  Parses the response into prose text + structured food card dicts
    8.  Updates the in-memory conversation history

The pipeline is designed to:
    - Be initialised ONCE per process (heavy resources loaded on startup)
    - Be called many times (query() is stateless except for memory)
    - Degrade gracefully — if any stage fails it returns a safe fallback
    - Work with any combination of embedding provider × vector store × LLM

Usage
─────
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()          # configure from .env / Settings
    pipeline.initialise()             # load vector store + embedding model

    result = pipeline.query(
        query   = "I want something warm",
        mood    = "cozy",
        filters = {"dietary": ["Vegan"], "cuisine": ["Italian"]},
        history = [{"role": "user", "content": "Hi"}, ...],
    )
    print(result["response"])         # LLM prose response
    print(result["recommendations"])  # list of food card dicts
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

Message      = dict[str, str]               # {"role": "user"|"assistant", "content": str}
Filters      = dict[str, list[str]]         # {"dietary": [...], "cuisine": [...]}
QueryResult  = dict[str, Any]               # {"response": str, "recommendations": list[dict]}
RetrievedDoc = dict[str, Any]               # {"document": str, "metadata": dict, "score": float}
FoodCard     = dict[str, Any]               # rendered card for the UI


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Mood-aware food recommendation RAG pipeline.

    Lifecycle
    ─────────
        pipeline = RAGPipeline()    # __init__: read config, no I/O
        pipeline.initialise()       # load embedding model + vector store
        result = pipeline.query()   # answer a user message

    Thread safety
    ─────────────
    The pipeline object is safe to share across Streamlit reruns via
    st.cache_resource because query() holds no mutable state between calls
    (conversation history is passed in, not stored on self).
    """

    def __init__(self, settings=None) -> None:
        """
        Initialise pipeline configuration.
        No heavy resources are loaded here — call initialise() first.

        Args:
            settings: Optional Settings instance. If None, reads from .env
                      via config.settings.get_settings().
        """
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()

        self.cfg            = settings
        self._embed_fn      = None      # kept for backward compat (not used)
        self._embedding_model = None    # rag.embeddings.EmbeddingModel instance
        self._vector_store  = None      # ChromaDB collection or FAISS dict
        self._llm_client    = None      # kept for backward compat (not used)
        self._llm           = None      # llm.base.LLMBase instance
        self._retriever     = None      # rag.retriever.Retriever instance
        self._initialised   = False

        logger.info(f"RAGPipeline created — {self.cfg.summary()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """
        Load all heavy resources:
            - Embedding model (OpenAI client or HuggingFace SentenceTransformer)
            - Vector store (ChromaDB persistent client or FAISS index)
            - LLM client (OpenAI / Anthropic / Google)

        Safe to call multiple times — subsequent calls are no-ops.

        Raises:
            RuntimeError: if required API keys are missing.
            FileNotFoundError: if the vector DB path does not exist.
        """
        if self._initialised:
            return

        t0 = time.perf_counter()
        logger.info("Initialising RAG pipeline …")

        from rag.embeddings import EmbeddingModel
        self._embedding_model = EmbeddingModel.from_settings(self.cfg)
        self._embed_fn        = self._embedding_model.embed_documents  # backward compat

        self._vector_store = self._build_vector_store()

        # LLM: key is NOT validated here — deferred to first complete() call.
        # This means the app starts and shows a clear "add your API key" message
        # in the chat instead of crashing with RuntimeError during initialise().
        from llms import LLMBase
        self._llm        = LLMBase.from_settings(self.cfg)
        self._llm_client = self._llm   # backward compat alias

        # Delegate all vector-search work to the standalone Retriever
        from rag.retriever import Retriever
        self._retriever    = Retriever(self._vector_store, self.cfg)

        self._initialised = True
        logger.info(
            f"RAG pipeline ready in {time.perf_counter() - t0:.2f}s  |  "
            f"llm={self.cfg.llm_provider.value}/{self.cfg.active_llm_model}  |  "
            f"embed={self.cfg.embedding_provider.value}  |  "
            f"vectors={self._vector_store.health().get('count', 0):,}"
        )

    def _require_init(self) -> None:
        if not self._initialised:
            raise RuntimeError(
                "RAGPipeline.initialise() must be called before query(). "
                "If using Streamlit, initialise inside @st.cache_resource."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Main query entry point
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        query:   str,
        mood:    Optional[str]   = None,
        filters: Optional[Filters] = None,
        history: Optional[list[Message]] = None,
    ) -> QueryResult:
        """
        Answer a user food query using RAG.

        Args:
            query:    The user's raw message, e.g. "something warm and hearty".
            mood:     Active mood key, e.g. "cozy". Used for query expansion
                      and prompt context. Pass None for mood-agnostic queries.
            filters:  Optional {"dietary": [...], "cuisine": [...]} filters.
                      Empty lists mean "no filter" (not "filter to nothing").
            history:  Last N conversation turns as [{"role", "content"}, ...].
                      Used to build a multi-turn LLM prompt.

        Returns:
            {
                "response":        str,          # LLM prose response
                "recommendations": list[dict],   # food cards for UI rendering
                "retrieved_docs":  list[dict],   # raw retrieval results (debug)
                "expanded_query":  str,           # query after mood expansion
                "mood":            str | None,
                "latency_ms":      float,
            }
        """
        self._require_init()
        t_start = time.perf_counter()
        filters = filters or {}
        history = history or []

        logger.debug(f"query='{query[:60]}…' mood={mood} filters={filters}")

        try:
            # ── Step 1: Expand query with mood descriptors ────────────────────
            expanded_query = self._expand_query(query, mood)
            logger.debug(f"Expanded query: '{expanded_query[:80]}…'")

            # ── Step 2: Embed the expanded query ──────────────────────────────
            query_vector = self._embed_texts([expanded_query])[0]

            # ── Step 3: Retrieve top-K documents ──────────────────────────────
            retrieved = self._retriever.retrieve(query_vector, mood, filters)
            logger.debug(f"Retrieved {len(retrieved)} documents.")

            # ── Step 4: Build LLM prompt ──────────────────────────────────────
            messages = self._build_messages(query, mood, retrieved, history)

            # ── Step 5: Call LLM ──────────────────────────────────────────────
            raw_response = self._call_llm(messages)

            # ── Step 6: Parse response into prose + food cards ────────────────
            prose, recommendations = self._parse_response(raw_response, retrieved)

            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(f"query() completed in {latency_ms:.0f}ms  docs={len(retrieved)}")

            return {
                "response":        prose,
                "recommendations": recommendations,
                "retrieved_docs":  retrieved,
                "expanded_query":  expanded_query,
                "mood":            mood,
                "latency_ms":      round(latency_ms, 1),
            }

        except Exception as exc:                                # noqa: BLE001
            logger.exception(f"RAG pipeline error: {exc}")
            return self._error_fallback(query, mood, exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Query expansion
    # ─────────────────────────────────────────────────────────────────────────

    def _expand_query(self, query: str, mood: Optional[str]) -> str:
        """
        Append mood descriptors to the raw query to improve retrieval.

        Example:
            query = "something warm"
            mood  = "cozy"
            →  "something warm. Mood: Cozy. warming hearty comforting snug..."
        """
        if not mood:
            return query

        from config.moods import build_expanded_query
        return build_expanded_query(query, mood)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Embedding
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings via the EmbeddingModel. Returns list of float vectors."""
        return self._embedding_model.embed_documents(texts)

    def _build_embedding_fn(self):
        """Return a callable: list[str] → list[list[float]]"""
        provider = self.cfg.embedding_provider.value

        if provider == "openai":
            return self._make_openai_embed_fn()
        elif provider == "huggingface":
            return self._make_hf_embed_fn()
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def _make_openai_embed_fn(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        if not self.cfg.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file, "
                "or switch to EMBEDDING_PROVIDER=huggingface for local embeddings."
            )

        client = OpenAI(api_key=self.cfg.openai_api_key)
        model  = self.cfg.embedding_model
        logger.info(f"  Embedding: OpenAI / {model}")

        def embed(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]

        return embed

    def _make_hf_embed_fn(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        model_name = self.cfg.hf_embedding_model
        logger.info(f"  Embedding: HuggingFace / {model_name} (loading…)")
        model = SentenceTransformer(model_name)
        logger.info("  HuggingFace embedding model loaded.")

        def embed(texts: list[str]) -> list[list[float]]:
            return [e.tolist() for e in model.encode(texts, show_progress_bar=False)]

        return embed

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_vector: list[float],
        mood:         Optional[str],
        filters:      Filters,
    ) -> list[RetrievedDoc]:
        """
        Query the vector store and return top-K results.

        Applies post-retrieval filtering:
            - Dietary tag filter (any of the user's dietary preferences must
              appear in the chunk's dietary_tags metadata field)
            - Cuisine filter (chunk's cuisine must be in the allowed list)
            - Similarity score threshold (drops weak matches)

        Returns at most cfg.retriever_top_k results, sorted by score desc.
        Always returns at least 1 result to avoid empty-response edge cases.
        """
        provider = self.cfg.vector_store_provider.value
        top_k    = self.cfg.retriever_top_k
        # Fetch extra to allow post-filtering headroom
        fetch_k  = max(top_k * 4, 20)

        if provider == "chroma":
            raw = self._retrieve_chroma(query_vector, fetch_k)
        elif provider == "faiss":
            raw = self._retrieve_faiss(query_vector, fetch_k)
        else:
            raise ValueError(f"Unknown vector store: {provider}")

        # ── Post-retrieval filtering ──────────────────────────────────────────
        filtered = self._apply_filters(raw, mood, filters)

        # ── Score threshold ───────────────────────────────────────────────────
        threshold = self.cfg.retriever_score_threshold
        filtered  = [d for d in filtered if d["score"] >= threshold]

        # ── Fallback: if filtering removed everything, use top raw results ────
        if not filtered and raw:
            logger.debug("All results filtered out — using top raw results as fallback.")
            filtered = raw[:top_k]

        return filtered[:top_k]

    def _retrieve_chroma(
        self,
        query_vector: list[float],
        fetch_k:      int,
    ) -> list[RetrievedDoc]:
        """Query ChromaDB and normalise results."""
        results = self._vector_store.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[RetrievedDoc] = []
        ids        = results.get("ids",        [[]])[0]
        documents  = results.get("documents",  [[]])[0]
        metadatas  = results.get("metadatas",  [[]])[0]
        distances  = results.get("distances",  [[]])[0]

        for i, (chunk_id, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # ChromaDB cosine distance → similarity: score = 1 - distance
            score = max(0.0, 1.0 - float(dist))
            docs.append({
                "chunk_id": chunk_id,
                "document": doc,
                "metadata": meta or {},
                "score":    round(score, 4),
                "rank":     i + 1,
            })

        return docs

    def _retrieve_faiss(
        self,
        query_vector: list[float],
        fetch_k:      int,
    ) -> list[RetrievedDoc]:
        """Query FAISS index and normalise results."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu numpy")

        index = self._vector_store["index"]
        meta  = self._vector_store["meta"]

        if index is None or index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return []

        vec = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(vec)

        fetch_k = min(fetch_k, index.ntotal)
        scores_arr, idx_arr = index.search(vec, fetch_k)
        scores = scores_arr[0].tolist()
        idxs   = idx_arr[0].tolist()

        docs: list[RetrievedDoc] = []
        for rank, (idx, score) in enumerate(zip(idxs, scores), start=1):
            if idx < 0 or idx >= len(meta):
                continue
            m = meta[idx]
            docs.append({
                "chunk_id": m.get("chunk_id", str(idx)),
                "document": "",     # FAISS doesn't store raw document text
                "metadata": m,
                "score":    round(max(0.0, float(score)), 4),
                "rank":     rank,
            })

        return docs

    def _apply_filters(
        self,
        docs:    list[RetrievedDoc],
        mood:    Optional[str],
        filters: Filters,
    ) -> list[RetrievedDoc]:
        """
        Apply dietary and cuisine post-retrieval metadata filters.

        Logic:
            dietary: if user specified any dietary tags, only keep docs
                     where AT LEAST ONE of those tags appears in the chunk's
                     dietary_tags field. (OR logic — more permissive)
            cuisine: if user specified cuisines, only keep docs whose
                     cuisine field is in the allowed list. (exact match)

        Mood-based soft filter:
            If a mood is active and fewer than half the docs match the mood's
            top cuisines, we skip the cuisine soft-filter to avoid over-restricting.
        """
        dietary_filter = [d.lower() for d in filters.get("dietary", [])]
        cuisine_filter = [c.lower() for c in filters.get("cuisine", [])]

        filtered = []
        for doc in docs:
            meta = doc["metadata"]

            # ── Dietary filter ────────────────────────────────────────────────
            if dietary_filter:
                doc_dietary = meta.get("dietary_tags", "").lower()
                if not any(tag in doc_dietary for tag in dietary_filter):
                    continue    # skip: none of user's dietary prefs match

            # ── Cuisine filter ────────────────────────────────────────────────
            if cuisine_filter:
                doc_cuisine = meta.get("cuisine", "").lower()
                if doc_cuisine not in cuisine_filter:
                    continue    # skip: cuisine not in user's selected list

            filtered.append(doc)

        return filtered

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Prompt builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        query:     str,
        mood:      Optional[str],
        retrieved: list[RetrievedDoc],
        history:   list[Message],
    ) -> list[Message]:
        """
        Build the full message array for the LLM:
            [system_message, ...history_messages, user_message]

        System message contains:
            - MoodBite persona and response format instructions
            - Mood context (from config/moods.py prompt_hint)
            - Retrieved food context (top-K chunks formatted as a menu)

        The retrieved context is injected into the system message (not the
        user message) so the LLM treats it as background knowledge, not as
        something the user said.
        """
        system_content = self._build_system_prompt(query, mood, retrieved)
        user_content   = self._build_user_message(query, mood)

        messages: list[Message] = [{"role": "system", "content": system_content}]

        # Trim history to configured memory limit
        limit   = self.cfg.conversation_memory_limit
        trimmed = history[-(limit * 2):]   # *2 because each turn = user + assistant
        messages.extend(trimmed)

        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_system_prompt(
        self,
        query:     str,
        mood:      Optional[str],
        retrieved: list[RetrievedDoc],
    ) -> str:
        """Build the system prompt string."""

        # ── Persona ───────────────────────────────────────────────────────────
        persona = (
            "You are MoodBite, a warm and knowledgeable food recommendation assistant. "
            "You suggest dishes that perfectly match how the user is feeling right now. "
            "You are enthusiastic about food, culturally informed, and genuinely caring. "
            "You never recommend food that contradicts the user's stated dietary restrictions. "
            "You always explain WHY a dish suits their current mood."
        )

        # ── Response format instructions ──────────────────────────────────────
        format_instructions = (
            "RESPONSE FORMAT INSTRUCTIONS:\n"
            "Your response must contain two clearly separated parts:\n\n"
            "PART 1 — PROSE RESPONSE:\n"
            "Write 2–4 warm, conversational sentences that:\n"
            "  - Acknowledge the user's mood (if provided)\n"
            "  - Introduce your top food recommendations naturally\n"
            "  - Briefly explain why these dishes suit how they're feeling\n"
            "  - End with a gentle follow-up question or invitation\n\n"
            "PART 2 — FOOD CARDS (JSON):\n"
            "After your prose, output a JSON block enclosed in ```json and ``` tags "
            "containing a list of 3–5 food recommendation objects. "
            "Each object must have these exact keys:\n"
            "  name         (string)  dish name\n"
            "  cuisine      (string)  cuisine type\n"
            "  description  (string)  1–2 sentence enticing description\n"
            "  tags         (array)   2–4 short tag strings e.g. ['Vegetarian', 'Comfort food']\n"
            "  score        (number)  relevance score 0.0–1.0\n"
            "  emoji        (string)  single relevant food emoji\n"
            "  prep_time    (string)  e.g. '25 min'\n"
            "  why_for_mood (string)  one sentence why this suits their mood\n\n"
            "IMPORTANT: Only recommend dishes from the food context provided below. "
            "Do not invent dishes not present in the context."
        )

        # ── Mood context ──────────────────────────────────────────────────────
        mood_section = ""
        if mood:
            from config.moods import get_mood_prompt_context, get_mood
            mood_hint = get_mood_prompt_context(mood)
            mood_obj  = get_mood(mood)
            avoid_str = (
                ", ".join(mood_obj.avoid) if mood_obj and mood_obj.avoid else "none"
            )
            mood_section = (
                f"\nCURRENT USER MOOD: {mood.upper()}\n"
                f"Mood guidance: {mood_hint}\n"
                f"Avoid recommending: {avoid_str}\n"
            )

        # ── Retrieved food context ────────────────────────────────────────────
        context_section = self._format_context(retrieved)

        return "\n\n".join(filter(None, [
            persona,
            format_instructions,
            mood_section,
            context_section,
        ]))

    def _format_context(self, retrieved: list[RetrievedDoc]) -> str:
        """Format retrieved documents as a numbered food menu for the prompt."""
        if not retrieved:
            return (
                "FOOD CONTEXT:\n"
                "No specific dishes were retrieved. Draw on your general food knowledge "
                "to make appropriate recommendations, but flag this to the user."
            )

        lines = ["FOOD CONTEXT (retrieved dishes — recommend only from these):"]
        for i, doc in enumerate(retrieved, start=1):
            meta = doc["metadata"]
            name        = meta.get("name",           "Unknown dish")
            cuisine     = meta.get("cuisine",        "")
            meal_type   = meta.get("meal_type",      "")
            description = meta.get("description",   doc.get("document", ""))[:200]
            ingredients = meta.get("ingredients",    "")
            moods       = meta.get("moods",          "")
            dietary     = meta.get("dietary_tags",   "none")
            spice       = meta.get("spice_level",    "")
            prep        = meta.get("prep_time_mins", "")
            calories    = meta.get("calories_approx","")
            score       = doc.get("score", 0.0)

            lines.append(
                f"\n[{i}] {name} ({cuisine} · {meal_type})\n"
                f"    Score: {score:.2f} | Prep: {prep} min | ~{calories} kcal\n"
                f"    Moods: {moods}\n"
                f"    Dietary: {dietary} | Spice: {spice}\n"
                f"    Ingredients: {ingredients[:120]}\n"
                f"    Description: {description}"
            )

        return "\n".join(lines)

    def _build_user_message(self, query: str, mood: Optional[str]) -> str:
        """Build the user turn content."""
        if mood:
            return f"I'm feeling {mood}. {query}"
        return query

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — LLM call
    # ─────────────────────────────────────────────────────────────────────────

    def _build_llm_client(self):
        """Build and return the LLM client for the configured provider."""
        provider = self.cfg.llm_provider.value

        if provider == "openai":
            return self._build_openai_client()
        elif provider == "anthropic":
            return self._build_anthropic_client()
        elif provider == "google":
            return self._build_google_client()
        else:
            raise ValueError(f"Unknown LLM provider: '{provider}'")

    def _build_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")
        if not self.cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        client = OpenAI(api_key=self.cfg.openai_api_key)
        logger.info(f"  LLM: OpenAI / {self.cfg.llm_model}")
        return client

    def _build_anthropic_client(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        if not self.cfg.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
        client = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key)
        logger.info(f"  LLM: Anthropic / {self.cfg.anthropic_model}")
        return client

    def _build_google_client(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")
        if not self.cfg.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env")
        genai.configure(api_key=self.cfg.google_api_key)
        model = genai.GenerativeModel(self.cfg.google_model)
        logger.info(f"  LLM: Google / {self.cfg.google_model}")
        return model

    def _call_llm(self, messages: list[Message]) -> str:
        """
        Delegate to the LLMBase instance (llm/ package).
        Retry logic lives inside LLMBase._call_with_retry().
        """
        return self._llm.complete(messages)

    def _call_openai(self, messages: list[Message]) -> str:
        response = self._llm_client.chat.completions.create(
            model=self.cfg.llm_model,
            messages=messages,
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.llm_max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, messages: list[Message]) -> str:
        # Anthropic separates system message from the conversation array
        system_msg   = ""
        conv_messages = []

        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                conv_messages.append(m)

        response = self._llm_client.messages.create(
            model=self.cfg.anthropic_model,
            max_tokens=self.cfg.llm_max_tokens,
            system=system_msg,
            messages=conv_messages,
        )
        return response.content[0].text if response.content else ""

    def _call_google(self, messages: list[Message]) -> str:
        # Google Gemini uses a different message format
        # Combine system + history into a single prompt string
        parts = []
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            if m["role"] == "system":
                parts.append(m["content"])
            else:
                parts.append(f"{role}: {m['content']}")

        full_prompt = "\n\n".join(parts)
        response = self._llm_client.generate_content(full_prompt)
        return response.text if response.text else ""

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 — Response parser
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_response(
        self,
        raw_response: str,
        retrieved:    list[RetrievedDoc],
    ) -> tuple[str, list[FoodCard]]:
        """
        Split the LLM response into:
            prose         str           Conversational text for the chat bubble
            food_cards    list[dict]    Structured dicts for the UI card grid

        The LLM is instructed to output:
            <prose text>
            ```json
            [{...}, ...]
            ```

        If the JSON block is missing or malformed, we synthesise food cards
        directly from the retrieved metadata so the UI always has something
        to render.
        """
        prose       = raw_response.strip()
        food_cards: list[FoodCard] = []

        # ── Try to extract ```json ... ``` block ──────────────────────────────
        json_match = re.search(
            r"```json\s*([\s\S]*?)\s*```",
            raw_response,
            re.IGNORECASE,
        )

        if json_match:
            json_str = json_match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    food_cards = [self._normalise_card(c) for c in parsed]
                    # Strip the JSON block from the prose
                    prose = raw_response[: json_match.start()].strip()
                    if not prose:
                        prose = self._generate_fallback_prose(food_cards)
            except json.JSONDecodeError as exc:
                logger.warning(f"JSON parse error in LLM response: {exc}")
                food_cards = []

        # ── Fallback: synthesise cards from retrieved metadata ────────────────
        if not food_cards and retrieved:
            logger.debug("Synthesising food cards from retrieved metadata.")
            food_cards = self._synthesise_cards_from_retrieved(retrieved)

        # ── Final prose sanitisation ──────────────────────────────────────────
        prose = self._sanitise_prose(prose)

        return prose, food_cards

    def _normalise_card(self, card: dict) -> FoodCard:
        """Ensure a food card dict has all required keys with safe defaults."""
        return {
            "name":         str(card.get("name",         "Unknown Dish")),
            "cuisine":      str(card.get("cuisine",       "")),
            "description":  str(card.get("description",   "")),
            "tags":         list(card.get("tags",          [])),
            "score":        float(card.get("score",        0.0)),
            "emoji":        str(card.get("emoji",          "🍽️")),
            "prep_time":    str(card.get("prep_time",      "")),
            "why_for_mood": str(card.get("why_for_mood",   "")),
        }

    def _synthesise_cards_from_retrieved(
        self,
        retrieved: list[RetrievedDoc],
    ) -> list[FoodCard]:
        """Build food card dicts directly from vector store metadata."""
        cards: list[FoodCard] = []
        for doc in retrieved:
            meta = doc["metadata"]
            cuisine  = meta.get("cuisine", "")
            tags: list[str] = []
            dietary = meta.get("dietary_tags", "")
            if dietary:
                tags.extend(dietary.split(",")[:2])
            meal_type = meta.get("meal_type", "")
            if meal_type:
                tags.append(meal_type)
            tags = [t.strip() for t in tags if t.strip()][:4]

            cards.append({
                "name":         meta.get("name",         "Unknown Dish"),
                "cuisine":      cuisine,
                "description":  meta.get("description",   "")[:200],
                "tags":         tags,
                "score":        doc.get("score",          0.0),
                "emoji":        _cuisine_emoji(cuisine),
                "prep_time":    f"{meta.get('prep_time_mins', '?')} min",
                "why_for_mood": "",
            })
        return cards

    def _generate_fallback_prose(self, cards: list[FoodCard]) -> str:
        """Generate a minimal prose response when the LLM omitted it."""
        if not cards:
            return "Here are some food recommendations for you."
        names = [c["name"] for c in cards[:3]]
        return (
            f"Based on how you're feeling, I'd suggest: "
            f"{', '.join(names)}. Each of these should hit the spot!"
        )

    def _sanitise_prose(self, prose: str) -> str:
        """Clean up the prose response — remove stray markdown artifacts."""
        # Remove any leftover ``` blocks
        prose = re.sub(r"```[a-z]*", "", prose)
        prose = re.sub(r"```",       "", prose)
        # Collapse excess blank lines
        prose = re.sub(r"\n{3,}", "\n\n", prose)
        return prose.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Vector store builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_vector_store(self):
        """Open the configured vector store via the vector_store package."""
        from vector_store import VectorStoreBase
        store = VectorStoreBase.from_settings(self.cfg)
        h = store.health()
        logger.info(
            f"  VectorStore: {h['provider']}  "
            f"count={h.get('count',0):,}  "
            f"ready={h['is_ready']}"
        )
        if not h["is_ready"]:
            logger.warning(
                "Vector store is empty. "
                "Run 'make ingest' to populate the vector database."
            )
        return store

    # ─────────────────────────────────────────────────────────────────────────
    # Error fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _error_fallback(
        self,
        query: str,
        mood:  Optional[str],
        exc:   Exception,
    ) -> QueryResult:
        """
        Return a graceful error response when the pipeline fails.
        Never raises — the Streamlit app must always get a result dict.
        """
        mood_str = f" (mood: {mood})" if mood else ""
        prose = (
            "I'm sorry — I ran into a small hiccup fetching your recommendations"
            f"{mood_str}. Please try again in a moment, or rephrase your question. "
            "I'm here to help!"
        )
        logger.error(f"Pipeline fallback triggered for query='{query[:40]}': {exc}")
        return {
            "response":        prose,
            "recommendations": [],
            "retrieved_docs":  [],
            "expanded_query":  query,
            "mood":            mood,
            "latency_ms":      0.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if initialise() has been called successfully."""
        return self._initialised

    def health_check(self) -> dict[str, Any]:
        """
        Return a dict describing the pipeline's health status.
        Used by admin/debug views.
        """
        retriever_health = (
            self._retriever.health() if self._retriever else {}
        )

        return {
            "initialised":      self._initialised,
            "llm_provider":     self.cfg.llm_provider.value,
            "llm_model":        self.cfg.active_llm_model,
            "llm_info":         self._llm.info() if self._llm else {},
            "embedding":        self.cfg.embedding_provider.value,
            "embedding_model":  self.cfg.active_embedding_model,
            "embedding_dim":    (
                self._embedding_model.dimension
                if self._embedding_model and self._embedding_model._dimension
                else None
            ),
            "vector_store":     self.cfg.vector_store_provider.value,
            "vector_count":     retriever_health.get("vector_count", 0),
            "retriever_ready":  retriever_health.get("is_ready", False),
            "top_k":            self.cfg.retriever_top_k,
            "score_threshold":  self.cfg.retriever_score_threshold,
            "debug_mode":       self.cfg.debug_mode,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

_CUISINE_EMOJI_MAP: dict[str, str] = {
    "italian": "🍝", "japanese": "🍣", "indian": "🍛", "mexican": "🌮",
    "chinese": "🥡", "thai": "🍜", "mediterranean": "🥗", "american": "🍔",
    "french": "🥐", "korean": "🍲", "middle eastern": "🧆", "spanish": "🥘",
    "greek": "🫒", "vietnamese": "🍜", "ethiopian": "🫓", "peruvian": "🐟",
    "turkish": "🥙", "moroccan": "🫕", "lebanese": "🧆", "brazilian": "🥩",
    "british": "🫖", "german": "🥨", "indonesian": "🍚", "filipino": "🍖",
    "caribbean": "🌴", "russian": "🥟", "polish": "🥣", "nigerian": "🍲",
    "argentinian": "🥩", "swedish": "🫙",
}


def _cuisine_emoji(cuisine: str) -> str:
    """Return an appropriate emoji for a cuisine string."""
    return _CUISINE_EMOJI_MAP.get(cuisine.lower().strip(), "🍽️")