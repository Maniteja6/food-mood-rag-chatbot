"""
rag/memory.py
═════════════
In-session conversation memory for the MoodBite RAG pipeline.

The ConversationMemory class stores the full dialogue for one user session
and provides trimmed slices to the PromptBuilder on each query turn.

Design notes
────────────
- Memory is stateful per-session and intentionally NOT shared across
  Streamlit reruns.  session_state.py stores the raw message list and passes
  it into pipeline.query() on every call — the pipeline and memory module
  stay stateless from Streamlit's perspective.
- This module is used by pipeline.py as an optional in-process buffer when
  the pipeline manages its own history (e.g. in scripts, tests, or CLI use).
  When running inside Streamlit, session_state.py owns the ground-truth list
  and passes it directly to pipeline.query(history=...).

Public API
──────────
    ConversationMemory(limit)
    memory.add_user(text, mood)
    memory.add_assistant(text, recommendations)
    memory.get_history(n)        → list[Message]
    memory.clear()
    memory.to_dict() / from_dict()
    len(memory)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# Type alias
Message = dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Turn dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """
    A single conversation turn — either user or assistant.

    role            "user" | "assistant"
    content         The message text
    timestamp       Unix float from time.time()
    mood            Active mood key at the time of this turn (may be None)
    recommendations List of food card dicts (assistant turns only)
    """
    role:            str
    content:         str
    timestamp:       float = field(default_factory=time.time)
    mood:            Optional[str] = None
    recommendations: list[dict]   = field(default_factory=list)

    def to_message(self) -> Message:
        """Return the minimal {role, content} dict for LLM calls."""
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict:
        """Return a fully serialisable dict (for JSON export)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", time.time()),
            mood=d.get("mood"),
            recommendations=d.get("recommendations", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ConversationMemory
# ─────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Manages in-session conversation history for a single user session.

    Parameters
    ──────────
    limit
        Maximum number of full turns (user + assistant pairs) to retain.
        Older turns are evicted when the limit is exceeded.
        Default: 10 (configurable via CONVERSATION_MEMORY_LIMIT in .env).

    Usage in pipeline (stateless Streamlit mode)
    ─────────────────────────────────────────────
        # Streamlit owns the history — ConversationMemory just wraps it
        memory = ConversationMemory.from_messages(session_state.messages)
        history_for_llm = memory.get_history()
        pipeline.query(history=history_for_llm, ...)

    Usage in scripts / tests (stateful mode)
    ─────────────────────────────────────────
        memory = ConversationMemory(limit=10)
        memory.add_user("I'm tired, what should I eat?", mood="tired")
        # ... call pipeline ...
        memory.add_assistant("Here's my suggestion …", recommendations=[...])
        history = memory.get_history()   # pass to next pipeline.query() call
    """

    def __init__(self, limit: int = 10) -> None:
        self._limit: int         = max(1, limit)
        self._turns: list[Turn]  = []

    # ─────────────────────────────────────────────────────────────────────────
    # Factory methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "ConversationMemory":
        """Build from the application Settings object."""
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        return cls(limit=settings.conversation_memory_limit)

    @classmethod
    def from_messages(
        cls,
        messages: list[dict],
        limit: int = 10,
    ) -> "ConversationMemory":
        """
        Build a ConversationMemory from a flat list of message dicts.

        Accepts both the minimal {role, content} format used by LLM APIs and
        the richer format stored in st.session_state (which may include
        timestamp, mood, and recommendations).

        Args:
            messages: List of message dicts from session_state.
            limit:    Memory limit (defaults to 10).

        Returns:
            Populated ConversationMemory instance.
        """
        mem = cls(limit=limit)
        for m in messages:
            turn = Turn(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", time.time()),
                mood=m.get("mood"),
                recommendations=m.get("recommendations", []),
            )
            mem._turns.append(turn)
        # Trim to limit immediately
        mem._evict()
        return mem

    # ─────────────────────────────────────────────────────────────────────────
    # Write methods
    # ─────────────────────────────────────────────────────────────────────────

    def add_user(self, text: str, mood: Optional[str] = None) -> None:
        """
        Append a user turn.

        Args:
            text: The user's message text.
            mood: The active mood key at the time (e.g. "cozy").
        """
        if not text or not text.strip():
            return
        self._turns.append(Turn(
            role="user",
            content=text.strip(),
            mood=mood,
        ))
        self._evict()

    def add_assistant(
        self,
        text:            str,
        recommendations: Optional[list[dict]] = None,
        mood:            Optional[str]        = None,
    ) -> None:
        """
        Append an assistant turn.

        Args:
            text:            The assistant's prose response.
            recommendations: Food card dicts attached to this response.
            mood:            The mood that was active when this was generated.
        """
        if not text or not text.strip():
            return
        self._turns.append(Turn(
            role="assistant",
            content=text.strip(),
            mood=mood,
            recommendations=recommendations or [],
        ))
        self._evict()

    def clear(self) -> None:
        """Remove all turns from memory."""
        self._turns.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Read methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_history(self, n: Optional[int] = None) -> list[Message]:
        """
        Return the last ``n`` turns as minimal {role, content} dicts,
        ready to pass into pipeline.query(history=...).

        Args:
            n: Number of turns to return.  Defaults to cfg memory limit.
               Pass None to get all stored turns.

        Returns:
            List of {"role": ..., "content": ...} dicts, oldest first.
        """
        if n is None:
            turns = self._turns
        else:
            turns = self._turns[-n:]
        return [t.to_message() for t in turns]

    def get_turns(self, n: Optional[int] = None) -> list[Turn]:
        """
        Return raw Turn objects (include mood, recommendations, timestamp).

        Useful for the debug panel and session export.
        """
        if n is None:
            return list(self._turns)
        return list(self._turns[-n:])

    def last_user_message(self) -> Optional[str]:
        """Return the content of the most recent user turn, or None."""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None

    def last_assistant_message(self) -> Optional[str]:
        """Return the content of the most recent assistant turn, or None."""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def last_mood(self) -> Optional[str]:
        """Return the most recently recorded mood across all turns."""
        for turn in reversed(self._turns):
            if turn.mood:
                return turn.mood
        return None

    def mood_history(self) -> list[str]:
        """Return all distinct moods seen in this session, in order."""
        seen: list[str] = []
        last: Optional[str] = None
        for turn in self._turns:
            if turn.mood and turn.mood != last:
                seen.append(turn.mood)
                last = turn.mood
        return seen

    def all_recommendations(self) -> list[dict]:
        """
        Return a flat list of all food cards recommended in this session.
        Useful for building a 'Your session history' summary view.
        """
        cards: list[dict] = []
        for turn in self._turns:
            if turn.role == "assistant":
                cards.extend(turn.recommendations)
        return cards

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise memory to a JSON-compatible dict.
        Used for session export and persistence (future feature).
        """
        return {
            "limit":  self._limit,
            "turns":  [t.to_dict() for t in self._turns],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        """Deserialise a memory dict produced by ``to_dict()``."""
        mem = cls(limit=data.get("limit", 10))
        mem._turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return mem

    # ─────────────────────────────────────────────────────────────────────────
    # Stats / utilities
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary dict for the sidebar metrics panel."""
        user_turns      = [t for t in self._turns if t.role == "user"]
        assistant_turns = [t for t in self._turns if t.role == "assistant"]
        total_recs      = sum(
            len(t.recommendations) for t in assistant_turns
        )
        return {
            "total_turns":      len(self._turns),
            "user_turns":       len(user_turns),
            "assistant_turns":  len(assistant_turns),
            "recommendations":  total_recs,
            "mood_history":     self.mood_history(),
            "last_mood":        self.last_mood(),
            "memory_limit":     self._limit,
        }

    def __len__(self) -> int:
        return len(self._turns)

    def __bool__(self) -> bool:
        return bool(self._turns)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"turns={len(self._turns)}, "
            f"limit={self._limit})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """
        Enforce the memory limit by removing the oldest turns.

        We evict in pairs (user + assistant) to avoid leaving orphaned turns.
        The eviction logic keeps the most recent ``limit`` complete turns.
        A complete turn = one user message + one assistant reply.
        We keep at most ``limit * 2`` individual Turn objects.
        """
        max_individual = self._limit * 2
        if len(self._turns) <= max_individual:
            return

        # Drop oldest turns
        self._turns = self._turns[-max_individual:]

        # Ensure we don't start with an orphaned assistant turn
        if self._turns and self._turns[0].role == "assistant":
            self._turns = self._turns[1:]