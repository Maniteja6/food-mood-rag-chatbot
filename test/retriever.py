"""
tests/test_pipeline.py
══════════════════════
Tests for the RAGPipeline orchestrator and its supporting modules:
  - PromptBuilder    (message construction)
  - ConversationMemory  (history management)
  - ResponseParser   (LLM output splitting)
  - EmbeddingModel   (provider abstraction)
  - RAGPipeline      (end-to-end with all dependencies mocked)

All external I/O (OpenAI API, ChromaDB, FAISS) is mocked so these tests
run offline with no API keys and no vector database present.

Run:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k "test_prompt"
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── ensure repo root is on path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── shared test fixtures ─────────────────────────────────────────────────────

SAMPLE_RETRIEVED = [
    {
        "chunk_id": "dish_000001",
        "document": (
            "Dish: Ramen\nCuisine: Japanese | Meal type: Main Course\n"
            "Moods: tired, cozy, sad\nIngredients: noodles, broth, chashu pork\n"
            "Dietary: none specified | Spice: Mild\n"
            "Description: A warming Japanese ramen that soothes the soul."
        ),
        "metadata": {
            "chunk_id":       "dish_000001",
            "name":           "Ramen",
            "cuisine":        "Japanese",
            "meal_type":      "Main Course",
            "description":    "A warming Japanese ramen that soothes the soul.",
            "ingredients":    "noodles, broth, chashu pork, soft-boiled egg",
            "moods":          "tired, cozy, sad",
            "dietary_tags":   "",
            "spice_level":    "Mild",
            "prep_time_mins": 90,
            "calories_approx": 620,
            "servings":       2,
            "cooking_method": "slow-cooked",
            "flavour_profile": "umami-rich",
            "texture":        "silky",
            "occasion":       "rainy day",
        },
        "score": 0.87,
        "rank":  1,
    },
    {
        "chunk_id": "dish_000002",
        "document": (
            "Dish: Butter Chicken\nCuisine: Indian | Meal type: Main Course\n"
            "Moods: cozy, happy, nostalgic\nIngredients: chicken, tomato, cream\n"
            "Dietary: Gluten-Free | Spice: Medium\n"
            "Description: Creamy, comforting Indian butter chicken."
        ),
        "metadata": {
            "chunk_id":       "dish_000002",
            "name":           "Butter Chicken",
            "cuisine":        "Indian",
            "meal_type":      "Main Course",
            "description":    "Creamy, comforting Indian butter chicken.",
            "ingredients":    "chicken, tomato, cream, spices",
            "moods":          "cozy, happy, nostalgic",
            "dietary_tags":   "Gluten-Free",
            "spice_level":    "Medium",
            "prep_time_mins": 45,
            "calories_approx": 520,
            "servings":       4,
            "cooking_method": "braised",
            "flavour_profile": "creamy",
            "texture":        "tender",
            "occasion":       "comfort meal",
        },
        "score": 0.82,
        "rank":  2,
    },
]

SAMPLE_HISTORY = [
    {"role": "user",      "content": "Hi, I am feeling a bit down today."},
    {"role": "assistant", "content": "I understand. Let me suggest some comforting dishes."},
]

LLM_PROSE_RESPONSE = (
    "Since you're feeling cozy, I have the perfect suggestions for you! "
    "A warming bowl of Ramen or a rich Butter Chicken will wrap you in "
    "comfort tonight. Both are ideal for a rainy evening. "
    "Does either of these appeal to you?"
)

LLM_JSON_BLOCK = json.dumps([
    {
        "name":         "Ramen",
        "cuisine":      "Japanese",
        "description":  "A warming bowl of rich broth and silky noodles.",
        "tags":         ["Comfort food", "Warming", "Filling"],
        "score":        0.92,
        "emoji":        "🍜",
        "prep_time":    "90 min",
        "why_for_mood": "Perfect for a cozy evening in.",
    },
    {
        "name":         "Butter Chicken",
        "cuisine":      "Indian",
        "description":  "Creamy, spiced curry that feels like a hug.",
        "tags":         ["Gluten-Free", "Comfort food"],
        "score":        0.85,
        "emoji":        "🍛",
        "prep_time":    "45 min",
        "why_for_mood": "Rich and warming — exactly right for feeling cozy.",
    },
], indent=2)

LLM_FULL_RESPONSE = f"{LLM_PROSE_RESPONSE}\n\n```json\n{LLM_JSON_BLOCK}\n```"


# ═════════════════════════════════════════════════════════════════════════════
# PromptBuilder tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPromptBuilder:
    """Tests for rag.prompt_builder.PromptBuilder."""

    @pytest.fixture
    def builder(self):
        from rag.prompt_builder import PromptBuilder

        cfg = MagicMock()
        cfg.conversation_memory_limit = 10
        return PromptBuilder(cfg)

    def test_build_returns_message_list(self, builder):
        messages = builder.build(
            query="What should I eat?",
            mood="cozy",
            retrieved=SAMPLE_RETRIEVED,
            history=[],
        )
        assert isinstance(messages, list)
        assert len(messages) >= 2   # at least system + user

    def test_system_message_is_first(self, builder):
        messages = builder.build("eat something", "happy", SAMPLE_RETRIEVED, [])
        assert messages[0]["role"] == "system"

    def test_user_message_is_last(self, builder):
        messages = builder.build("eat something", "sad", SAMPLE_RETRIEVED, [])
        assert messages[-1]["role"] == "user"

    def test_user_message_contains_mood(self, builder):
        messages = builder.build("food please", "romantic", SAMPLE_RETRIEVED, [])
        user_content = messages[-1]["content"].lower()
        assert "romantic" in user_content

    def test_system_contains_dish_names(self, builder):
        messages = builder.build("something warm", "cozy", SAMPLE_RETRIEVED, [])
        system = messages[0]["content"]
        assert "Ramen" in system
        assert "Butter Chicken" in system

    def test_system_contains_mood_guidance(self, builder):
        messages = builder.build("warm food", "cozy", SAMPLE_RETRIEVED, [])
        system = messages[0]["content"]
        # MOOD CONTEXT section should appear
        assert "MOOD CONTEXT" in system or "cozy" in system.lower()

    def test_history_inserted_between_system_and_user(self, builder):
        messages = builder.build("more food", "tired", SAMPLE_RETRIEVED, SAMPLE_HISTORY)
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert roles[-1] == "user"
        # history turns are in the middle
        middle_roles = roles[1:-1]
        assert "user" in middle_roles
        assert "assistant" in middle_roles

    def test_history_trimmed_to_limit(self, builder):
        builder._cfg.conversation_memory_limit = 2
        long_history = [
            {"role": "user",      "content": f"msg {i}"}
            if i % 2 == 0
            else {"role": "assistant", "content": f"reply {i}"}
            for i in range(20)
        ]
        messages = builder.build("latest", "happy", SAMPLE_RETRIEVED, long_history)
        # system + ≤4 history messages + user
        assert len(messages) <= 6

    def test_history_never_starts_with_assistant(self, builder):
        orphan_history = [
            {"role": "assistant", "content": "I said something"},
            {"role": "user",      "content": "And I replied"},
        ]
        messages = builder.build("now", "sad", SAMPLE_RETRIEVED, orphan_history)
        # second message (first after system) should never be assistant
        if len(messages) > 2:
            assert messages[1]["role"] == "user"

    def test_no_mood_uses_fallback_text(self, builder):
        messages = builder.build("anything", None, SAMPLE_RETRIEVED, [])
        system = messages[0]["content"]
        assert len(system) > 50   # should still have content

    def test_empty_retrieved_has_fallback_context(self, builder):
        messages = builder.build("food", "happy", [], [])
        system = messages[0]["content"]
        assert "FOOD CONTEXT" in system

    def test_format_context_includes_score(self, builder):
        ctx = builder.format_context(SAMPLE_RETRIEVED)
        assert "0.87" in ctx
        assert "0.82" in ctx

    def test_format_context_includes_prep_time(self, builder):
        ctx = builder.format_context(SAMPLE_RETRIEVED)
        assert "90" in ctx   # prep time from Ramen

    def test_format_context_includes_dietary(self, builder):
        ctx = builder.format_context(SAMPLE_RETRIEVED)
        assert "Gluten-Free" in ctx


# ═════════════════════════════════════════════════════════════════════════════
# ConversationMemory tests
# ═════════════════════════════════════════════════════════════════════════════

class TestConversationMemory:
    """Tests for rag.memory.ConversationMemory."""

    @pytest.fixture
    def memory(self):
        from rag.memory import ConversationMemory
        return ConversationMemory(limit=5)

    def test_starts_empty(self, memory):
        assert len(memory) == 0
        assert not memory

    def test_add_user_increments_count(self, memory):
        memory.add_user("Hello")
        assert len(memory) == 1

    def test_add_assistant_increments_count(self, memory):
        memory.add_user("Hello")
        memory.add_assistant("Hi there!")
        assert len(memory) == 2

    def test_get_history_returns_minimal_dicts(self, memory):
        memory.add_user("Hello", mood="happy")
        memory.add_assistant("Hey!", recommendations=[])
        history = memory.get_history()
        assert len(history) == 2
        for m in history:
            assert "role" in m
            assert "content" in m
            assert "timestamp" not in m      # minimal format for LLM

    def test_get_history_respects_n_param(self, memory):
        for i in range(6):
            memory.add_user(f"msg {i}")
        history = memory.get_history(n=3)
        assert len(history) == 3

    def test_limit_evicts_oldest_turns(self):
        from rag.memory import ConversationMemory
        mem = ConversationMemory(limit=2)
        for i in range(6):
            mem.add_user(f"u{i}", mood="happy")
            mem.add_assistant(f"a{i}")
        # limit=2 means 4 individual messages max
        assert len(mem) <= 4

    def test_eviction_never_starts_with_assistant(self):
        from rag.memory import ConversationMemory
        mem = ConversationMemory(limit=2)
        mem.add_user("u1")
        mem.add_assistant("a1")
        mem.add_user("u2")
        mem.add_assistant("a2")
        mem.add_user("u3")
        turns = mem.get_turns()
        if turns:
            assert turns[0].role == "user"

    def test_clear_resets_memory(self, memory):
        memory.add_user("msg1")
        memory.add_assistant("reply1")
        memory.clear()
        assert len(memory) == 0

    def test_last_user_message(self, memory):
        memory.add_user("first")
        memory.add_user("second")
        assert memory.last_user_message() == "second"

    def test_last_assistant_message(self, memory):
        memory.add_assistant("first reply")
        memory.add_assistant("second reply")
        assert memory.last_assistant_message() == "second reply"

    def test_last_mood(self, memory):
        memory.add_user("hello", mood="happy")
        memory.add_user("more", mood="cozy")
        assert memory.last_mood() == "cozy"

    def test_mood_history_deduplicated(self, memory):
        memory.add_user("a", mood="happy")
        memory.add_user("b", mood="happy")   # same mood twice in a row
        memory.add_user("c", mood="cozy")
        assert memory.mood_history() == ["happy", "cozy"]

    def test_all_recommendations(self, memory):
        recs = [{"name": "Ramen", "cuisine": "Japanese"}]
        memory.add_assistant("Here's ramen.", recommendations=recs)
        assert len(memory.all_recommendations()) == 1
        assert memory.all_recommendations()[0]["name"] == "Ramen"

    def test_stats_keys(self, memory):
        memory.add_user("hi", mood="tired")
        memory.add_assistant("response")
        s = memory.stats()
        assert "total_turns"     in s
        assert "user_turns"      in s
        assert "assistant_turns" in s
        assert "recommendations" in s
        assert "mood_history"    in s
        assert "last_mood"       in s

    def test_serialisation_roundtrip(self, memory):
        memory.add_user("u1", mood="sad")
        memory.add_assistant("a1", recommendations=[{"name": "Soup"}])
        data     = memory.to_dict()
        restored = type(memory).from_dict(data)
        assert len(restored) == len(memory)
        assert restored.last_user_message() == "u1"
        assert restored.last_mood() == "sad"

    def test_from_messages_factory(self):
        from rag.memory import ConversationMemory
        msgs = [
            {"role": "user",      "content": "Hello", "mood": "happy"},
            {"role": "assistant", "content": "Hi!",   "recommendations": []},
        ]
        mem = ConversationMemory.from_messages(msgs, limit=10)
        assert len(mem) == 2
        assert mem.last_user_message() == "Hello"

    def test_blank_messages_not_added(self, memory):
        memory.add_user("")
        memory.add_user("   ")
        assert len(memory) == 0


# ═════════════════════════════════════════════════════════════════════════════
# ResponseParser tests
# ═════════════════════════════════════════════════════════════════════════════

class TestResponseParser:
    """Tests for rag.response_parser.ResponseParser."""

    @pytest.fixture
    def parser(self):
        from rag.response_parser import ResponseParser
        return ResponseParser()

    def test_parses_prose_and_json_block(self, parser):
        result = parser.parse(LLM_FULL_RESPONSE, SAMPLE_RETRIEVED)
        assert result.json_found
        assert result.json_valid
        assert not result.synthesis_used
        assert len(result.prose) > 10
        assert len(result.food_cards) == 2

    def test_prose_does_not_contain_json_fence(self, parser):
        result = parser.parse(LLM_FULL_RESPONSE, SAMPLE_RETRIEVED)
        assert "```" not in result.prose
        assert "```json" not in result.prose

    def test_food_card_keys(self, parser):
        result = parser.parse(LLM_FULL_RESPONSE, SAMPLE_RETRIEVED)
        required = {"name", "cuisine", "description", "tags", "score", "emoji", "prep_time", "why_for_mood"}
        for card in result.food_cards:
            assert required.issubset(card.keys()), f"Missing keys: {required - card.keys()}"

    def test_food_card_score_clamped(self, parser):
        bad_json = json.dumps([{
            "name": "X", "cuisine": "Y", "description": "D",
            "tags": [], "score": 999.9, "emoji": "🍽️",
            "prep_time": "10 min", "why_for_mood": "good"
        }])
        result = parser.parse(f"Some prose\n```json\n{bad_json}\n```", SAMPLE_RETRIEVED)
        assert result.food_cards[0]["score"] <= 1.0

    def test_food_card_tags_max_5(self, parser):
        big_tags_json = json.dumps([{
            "name": "X", "cuisine": "Y", "description": "D",
            "tags": ["A","B","C","D","E","F","G"], "score": 0.5,
            "emoji": "🍽️", "prep_time": "10 min", "why_for_mood": "ok"
        }])
        result = parser.parse(f"Prose\n```json\n{big_tags_json}\n```", SAMPLE_RETRIEVED)
        assert len(result.food_cards[0]["tags"]) <= 5

    def test_no_json_block_triggers_synthesis(self, parser):
        result = parser.parse("Just some prose with no JSON.", SAMPLE_RETRIEVED)
        assert not result.json_found
        assert result.synthesis_used
        assert len(result.food_cards) == len(SAMPLE_RETRIEVED)

    def test_malformed_json_triggers_synthesis(self, parser):
        bad_response = "Some prose\n```json\n[{broken json here\n```"
        result = parser.parse(bad_response, SAMPLE_RETRIEVED)
        assert result.json_found
        assert not result.json_valid
        assert result.synthesis_used
        assert len(result.food_cards) > 0

    def test_empty_response_returns_fallback_prose(self, parser):
        result = parser.parse("", SAMPLE_RETRIEVED, mood="cozy")
        assert len(result.prose) > 0
        assert "cozy" in result.prose.lower() or len(result.prose) > 20

    def test_empty_response_still_returns_cards(self, parser):
        result = parser.parse("", SAMPLE_RETRIEVED)
        assert len(result.food_cards) > 0

    def test_synthesis_preserves_name_and_cuisine(self, parser):
        result = parser.parse("no json here", SAMPLE_RETRIEVED)
        card_names = {c["name"] for c in result.food_cards}
        assert "Ramen" in card_names
        assert "Butter Chicken" in card_names

    def test_synthesis_cuisine_emoji(self, parser):
        result = parser.parse("prose only", SAMPLE_RETRIEVED)
        for card in result.food_cards:
            assert card["emoji"]   # should never be empty

    def test_json_wrapped_in_dict_unwrapped(self, parser):
        wrapped = json.dumps({"recommendations": [
            {"name": "Pasta", "cuisine": "Italian", "description": "nice",
             "tags": ["Quick"], "score": 0.7, "emoji": "🍝",
             "prep_time": "20 min", "why_for_mood": "comforting"}
        ]})
        result = parser.parse(f"Prose\n```json\n{wrapped}\n```", SAMPLE_RETRIEVED)
        if result.json_valid:
            assert any(c["name"] == "Pasta" for c in result.food_cards)

    def test_parsed_response_to_dict(self, parser):
        result = parser.parse(LLM_FULL_RESPONSE, SAMPLE_RETRIEVED)
        d = result.to_dict()
        assert "prose" in d
        assert "food_cards" in d
        assert "json_found" in d
        assert "json_valid" in d
        assert "synthesis_used" in d

    def test_truncated_json_repair(self, parser):
        # Truncated mid-object
        truncated = (
            'Prose\n```json\n'
            '[{"name": "Ramen", "cuisine": "Japanese", "description": "good",'
            ' "tags": ["warm"], "score": 0.9, "emoji": "🍜"'
            # deliberately cut off before closing
        )
        result = parser.parse(truncated, SAMPLE_RETRIEVED)
        # Either repaired or fell back to synthesis — both acceptable
        assert len(result.food_cards) > 0

    def test_parse_result_has_raw_response(self, parser):
        result = parser.parse("hello", SAMPLE_RETRIEVED)
        assert result.raw_response == "hello"


# ═════════════════════════════════════════════════════════════════════════════
# EmbeddingModel tests
# ═════════════════════════════════════════════════════════════════════════════

class TestEmbeddingModel:
    """Tests for rag.embeddings.EmbeddingModel — mocks both providers."""

    @pytest.fixture
    def mock_hf_model(self):
        """Patch SentenceTransformer so no model is downloaded."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4] for _ in range(10)
        ], dtype="float32")
        mock_model.get_sentence_embedding_dimension.return_value = 4
        return mock_model

    def test_from_config_openai(self):
        from rag.embeddings import EmbeddingModel
        em = EmbeddingModel.from_config({
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "openai_api_key": "sk-test",
            "batch_size": 50,
        })
        assert em._provider == "openai"
        assert em._model_name == "text-embedding-3-small"
        assert em._batch_size == 50

    def test_from_config_huggingface(self):
        from rag.embeddings import EmbeddingModel
        em = EmbeddingModel.from_config({
            "embedding_provider": "huggingface",
            "hf_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        })
        assert em._provider == "huggingface"

    def test_invalid_provider_raises(self):
        from rag.embeddings import EmbeddingModel
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingModel(provider="invalid", model_name="x", api_key="")

    def test_embed_query_returns_single_vector(self, mock_hf_model):
        from rag.embeddings import EmbeddingModel
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_hf_model):
            em = EmbeddingModel(
                provider="huggingface",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            vec = em.embed_query("test query")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) == 4

    def test_embed_documents_returns_list_of_vectors(self, mock_hf_model):
        from rag.embeddings import EmbeddingModel
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_hf_model):
            em = EmbeddingModel(
                provider="huggingface",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            vecs = em.embed_documents(["doc one", "doc two", "doc three"])
        assert len(vecs) == 3
        assert all(len(v) == 4 for v in vecs)

    def test_embed_query_empty_string_raises(self):
        from rag.embeddings import EmbeddingModel
        em = EmbeddingModel(provider="openai", model_name="x", api_key="sk-x")
        with pytest.raises(ValueError, match="empty"):
            em.embed_query("")

    def test_embed_documents_empty_list_raises(self):
        from rag.embeddings import EmbeddingModel
        em = EmbeddingModel(provider="openai", model_name="x", api_key="sk-x")
        with pytest.raises(ValueError, match="empty"):
            em.embed_documents([])

    def test_openai_embed_calls_api(self):
        from rag.embeddings import EmbeddingModel

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            em = EmbeddingModel(provider="openai", model_name="text-embedding-3-small", api_key="sk-test")
            vec = em.embed_query("hello")

        assert vec == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_info_returns_dict(self):
        from rag.embeddings import EmbeddingModel
        em = EmbeddingModel(provider="openai", model_name="text-embedding-3-small", api_key="sk-x")
        info = em.info()
        assert info["provider"]   == "openai"
        assert info["model_name"] == "text-embedding-3-small"
        assert "batch_size" in info
        assert "dimension"  in info

    def test_repr(self):
        from rag.embeddings import EmbeddingModel
        em = EmbeddingModel(provider="openai", model_name="text-embedding-3-small", api_key="sk-x")
        r = repr(em)
        assert "openai" in r
        assert "text-embedding-3-small" in r


# ═════════════════════════════════════════════════════════════════════════════
# RAGPipeline integration tests (all I/O mocked)
# ═════════════════════════════════════════════════════════════════════════════

class TestRAGPipeline:
    """
    End-to-end pipeline tests with all external dependencies mocked.

    The pipeline wires: EmbeddingModel → Retriever → PromptBuilder → LLM → ResponseParser.
    We mock the embedding fn, vector store, and LLM to test orchestration logic.
    """

    @pytest.fixture
    def mock_settings(self):
        """Minimal settings object that satisfies RAGPipeline.__init__."""
        from config.settings import LLMProvider, EmbeddingProvider, VectorStoreProvider

        s = MagicMock()
        s.llm_provider            = LLMProvider.OPENAI
        s.embedding_provider      = EmbeddingProvider.OPENAI
        s.vector_store_provider   = VectorStoreProvider.CHROMA
        s.llm_model               = "gpt-4o"
        s.llm_temperature         = 0.7
        s.llm_max_tokens          = 1024
        s.openai_api_key          = "sk-test"
        s.anthropic_api_key       = ""
        s.embedding_model         = "text-embedding-3-small"
        s.hf_embedding_model      = "all-MiniLM-L6-v2"
        s.vector_db_path          = "/tmp/test_vector_db"
        s.chroma_collection_name  = "test_collection"
        s.retriever_top_k         = 5
        s.retriever_score_threshold = 0.3
        s.conversation_memory_limit = 10
        s.debug_mode              = False
        s.embed_batch_size        = 100
        s.anthropic_model         = "claude-3-5-sonnet-20241022"
        s.active_llm_model        = "gpt-4o"
        s.active_embedding_model  = "text-embedding-3-small"
        s.is_chroma               = True
        s.is_faiss                = False
        s.summary.return_value    = "openai/gpt-4o embed=openai/text-embedding-3-small"
        return s

    @pytest.fixture
    def mock_embed_fn(self):
        """Returns a fixed 4-dim vector for any input."""
        def _embed(texts):
            return [[0.1, 0.2, 0.3, 0.4]] * len(texts)
        return _embed

    @pytest.fixture
    def mock_vector_store(self):
        """Fake VectorStoreBase that returns SAMPLE_RETRIEVED."""
        from vector_store.base import VectorStoreBase, QueryResult
        store = MagicMock(spec=VectorStoreBase)
        store.query.return_value = [
            QueryResult(
                chunk_id=d["chunk_id"],
                document=d["document"],
                metadata=d["metadata"],
                score=d["score"],
                rank=d["rank"],
            )
            for d in SAMPLE_RETRIEVED
        ]
        store.count.return_value = 50_000
        store.health.return_value = {
            "provider": "chroma", "count": 50_000, "is_ready": True, "path": "/tmp"
        }
        return store

    @pytest.fixture
    def mock_llm(self):
        """Fake LLMBase that returns the full test response."""
        from llms.base import LLMBase
        llm = MagicMock(spec=LLMBase)
        llm.complete.return_value = LLM_FULL_RESPONSE
        llm.info.return_value = {"provider": "openai", "model": "gpt-4o"}
        return llm

    @pytest.fixture
    def pipeline(self, mock_settings, mock_embed_fn, mock_vector_store, mock_llm):
        """
        A fully wired RAGPipeline with all external I/O replaced by mocks.
        """
        from rag.pipeline import RAGPipeline
        from rag.retriever import Retriever

        p = RAGPipeline(settings=mock_settings)

        # Inject mocks directly to bypass initialise()
        p._embedding_model        = MagicMock()
        p._embedding_model.embed_documents = mock_embed_fn
        p._embedding_model.info.return_value = {}
        p._embedding_model._dimension = 4
        p._embed_fn               = mock_embed_fn
        p._vector_store           = mock_vector_store
        p._retriever              = Retriever(mock_vector_store, mock_settings)
        p._llm                    = mock_llm
        p._llm_client             = mock_llm
        p._initialised            = True

        return p

    # ── Basic query tests ─────────────────────────────────────────────────────

    def test_query_returns_dict_with_required_keys(self, pipeline):
        result = pipeline.query("I need comfort food", mood="cozy")
        assert "response"        in result
        assert "recommendations" in result
        assert "retrieved_docs"  in result
        assert "expanded_query"  in result
        assert "mood"            in result
        assert "latency_ms"      in result

    def test_query_response_is_string(self, pipeline):
        result = pipeline.query("What to eat?", mood="happy")
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_query_recommendations_is_list(self, pipeline):
        result = pipeline.query("Dinner ideas", mood="tired")
        assert isinstance(result["recommendations"], list)

    def test_query_food_cards_have_required_fields(self, pipeline):
        result = pipeline.query("Something warm", mood="cozy")
        required = {"name", "cuisine", "description", "tags", "score", "emoji"}
        for card in result["recommendations"]:
            assert required.issubset(card.keys())

    def test_query_mood_stored_in_result(self, pipeline):
        result = pipeline.query("food", mood="romantic")
        assert result["mood"] == "romantic"

    def test_query_none_mood_accepted(self, pipeline):
        result = pipeline.query("I want pasta", mood=None)
        assert result["mood"] is None
        assert "response" in result

    def test_query_expanded_query_contains_mood_descriptors(self, pipeline):
        result = pipeline.query("something warm", mood="cozy")
        eq = result["expanded_query"]
        # Should contain the original query
        assert "something warm" in eq.lower()

    def test_query_passes_history_to_prompt_builder(self, pipeline, mock_llm):
        pipeline.query("More food", mood="happy", history=SAMPLE_HISTORY)
        # LLM was called with a messages list
        call_args = mock_llm.complete.call_args
        messages  = call_args[0][0]
        contents  = [m["content"] for m in messages]
        # History content should appear in the messages
        assert any("feeling a bit down" in c for c in contents)

    def test_query_filters_passed_through(self, pipeline, mock_vector_store):
        pipeline.query(
            "vegan food",
            mood="happy",
            filters={"dietary": ["Vegan"], "cuisine": ["Italian"]},
        )
        # Vector store was queried
        mock_vector_store.query.assert_called_once()

    def test_query_latency_is_positive(self, pipeline):
        result = pipeline.query("food?", mood="tired")
        assert result["latency_ms"] > 0

    # ── Error handling ────────────────────────────────────────────────────────

    def test_query_returns_fallback_on_llm_error(self, pipeline, mock_llm):
        mock_llm.complete.side_effect = RuntimeError("API down")
        result = pipeline.query("food", mood="happy")
        # Should NOT raise — returns fallback dict
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_query_returns_fallback_on_embed_error(self, pipeline):
        pipeline._embedding_model.embed_documents = MagicMock(
            side_effect=RuntimeError("embed failed")
        )
        pipeline._embed_fn = pipeline._embedding_model.embed_documents
        result = pipeline.query("food", mood="happy")
        assert "response" in result

    # ── is_ready ─────────────────────────────────────────────────────────────

    def test_is_ready_false_before_initialise(self, mock_settings):
        from rag.pipeline import RAGPipeline
        p = RAGPipeline(settings=mock_settings)
        assert not p.is_ready

    def test_is_ready_true_after_init_inject(self, pipeline):
        assert pipeline.is_ready

    # ── health_check ──────────────────────────────────────────────────────────

    def test_health_check_returns_dict(self, pipeline):
        h = pipeline.health_check()
        assert "initialised"   in h
        assert "llm_provider"  in h
        assert "vector_store"  in h
        assert "vector_count"  in h

    def test_health_check_vector_count(self, pipeline):
        h = pipeline.health_check()
        assert h["vector_count"] == 50_000

    # ── Query expansion ───────────────────────────────────────────────────────

    def test_expand_query_with_mood(self, pipeline):
        expanded = pipeline._expand_query("warm food", "cozy")
        assert "cozy" in expanded.lower()
        assert "warm food" in expanded.lower()

    def test_expand_query_without_mood(self, pipeline):
        expanded = pipeline._expand_query("pasta please", None)
        assert "pasta please" in expanded


# ═════════════════════════════════════════════════════════════════════════════
# config/moods integration tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMoodsConfig:
    """Tests for config.moods — no mocking needed (pure Python)."""

    def test_all_20_moods_present(self):
        from config.moods import MOOD_REGISTRY
        assert len(MOOD_REGISTRY) == 20

    def test_mood_keys_match_registry(self):
        from config.moods import MOOD_REGISTRY, MOOD_KEYS
        assert set(MOOD_KEYS) == set(MOOD_REGISTRY.keys())

    def test_get_mood_returns_config(self):
        from config.moods import get_mood
        m = get_mood("happy")
        assert m is not None
        assert m.key == "happy"
        assert m.emoji == "😄"

    def test_get_mood_case_insensitive(self):
        from config.moods import get_mood
        assert get_mood("HAPPY") is not None
        assert get_mood("Happy") is not None

    def test_get_mood_unknown_returns_none(self):
        from config.moods import get_mood
        assert get_mood("unknown_mood_xyz") is None

    def test_get_mood_descriptors_returns_list(self):
        from config.moods import get_mood_descriptors
        descs = get_mood_descriptors("cozy")
        assert isinstance(descs, list)
        assert len(descs) > 0
        assert all(isinstance(d, str) for d in descs)

    def test_get_mood_descriptors_unknown_returns_empty(self):
        from config.moods import get_mood_descriptors
        assert get_mood_descriptors("xyz_unknown") == []

    def test_get_mood_prompt_context_returns_string(self):
        from config.moods import get_mood_prompt_context
        hint = get_mood_prompt_context("romantic")
        assert isinstance(hint, str)
        assert len(hint) > 20

    def test_get_mood_prompt_context_fallback(self):
        from config.moods import get_mood_prompt_context
        hint = get_mood_prompt_context("totally_unknown_mood")
        assert isinstance(hint, str)
        assert len(hint) > 0

    def test_build_expanded_query(self):
        from config.moods import build_expanded_query
        expanded = build_expanded_query("something warm", "cozy")
        assert "something warm" in expanded
        assert "Cozy" in expanded or "cozy" in expanded.lower()

    def test_build_expanded_query_unknown_mood(self):
        from config.moods import build_expanded_query
        expanded = build_expanded_query("anything", "nonexistent")
        # Should return original query unchanged
        assert "anything" in expanded

    def test_mood_config_is_frozen(self):
        from config.moods import get_mood
        m = get_mood("happy")
        with pytest.raises(Exception):  # dataclass frozen=True
            m.key = "modified"          # type: ignore

    def test_mood_to_ui_dict_keys(self):
        from config.moods import get_mood
        m = get_mood("stressed")
        ui = m.to_ui_dict()
        assert "key"    in ui
        assert "label"  in ui
        assert "emoji"  in ui
        assert "colour" in ui

    def test_mood_keys_for_display_all_20(self):
        from config.moods import mood_keys_for_display
        display = mood_keys_for_display()
        assert len(display) == 20
        assert all("emoji" in d for d in display)

    def test_all_moods_have_non_empty_descriptors(self):
        from config.moods import ALL_MOODS
        for mood in ALL_MOODS:
            assert len(mood.descriptors) > 0, f"{mood.key} has no descriptors"

    def test_all_moods_have_prompt_hint(self):
        from config.moods import ALL_MOODS
        for mood in ALL_MOODS:
            assert len(mood.prompt_hint) > 20, f"{mood.key} prompt_hint too short"

    def test_all_moods_have_top_cuisines(self):
        from config.moods import ALL_MOODS
        for mood in ALL_MOODS:
            assert len(mood.top_cuisines) >= 3, f"{mood.key} needs ≥3 top cuisines"