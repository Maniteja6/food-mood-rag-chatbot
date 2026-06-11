"""
session_state.py — Streamlit Session State Manager

Centralises all st.session_state initialisation and mutation so that
main.py and ui_components.py never touch st.session_state keys directly.

Keys managed:
    messages          list[dict]   Full chat history (role + content + metadata)
    current_mood      str | None   Active mood selected by the user
    mood_history      list[str]    All moods used in this session
    rag_pipeline      object|None  Initialised RAG pipeline (cached across reruns)
    recommendations   list[dict]   Most recent food recommendation cards
    is_loading        bool         True while the LLM is generating a response
    session_id        str          Unique ID for this browser session
    dietary_filters   list[str]    Active dietary preference filters
    cuisine_filters   list[str]    Active cuisine filters
    message_count     int          Total messages sent (for sidebar metrics)
    last_query_time   float|None   Timestamp of last query (for rate limiting UI)
"""

from __future__ import annotations

import uuid
import time
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "messages": [],
    "current_mood": None,
    "mood_history": [],
    "rag_pipeline": None,
    "recommendations": [],
    "is_loading": False,
    "session_id": None,        # initialised with a real UUID in init()
    "dietary_filters": [],
    "cuisine_filters": [],
    "message_count": 0,
    "last_query_time": None,
}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init() -> None:
    """
    Call once at the top of main.py.
    Seeds st.session_state with default values for any key that doesn't
    already exist (safe to call on every rerun).
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Session ID is a one-time UUID — set it only if truly absent.
    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def add_user_message(content: str) -> None:
    """Append a user message to the chat history."""
    st.session_state.messages.append({
        "role": "user",
        "content": content,
        "timestamp": time.time(),
        "mood": st.session_state.current_mood,
    })
    st.session_state.message_count += 1
    st.session_state.last_query_time = time.time()


def add_assistant_message(
    content: str,
    recommendations: list[dict] | None = None,
) -> None:
    """
    Append an assistant message to the chat history.

    Args:
        content:         The markdown text response from the LLM.
        recommendations: Optional list of food card dicts attached to this turn.
    """
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "timestamp": time.time(),
        "mood": st.session_state.current_mood,
        "recommendations": recommendations or [],
    })
    if recommendations:
        st.session_state.recommendations = recommendations


def get_messages() -> list[dict]:
    """Return the full chat history."""
    return st.session_state.messages


def clear_messages() -> None:
    """Wipe the chat history (keeps mood and filters)."""
    st.session_state.messages = []
    st.session_state.recommendations = []
    st.session_state.message_count = 0


def get_conversation_for_rag(limit: int = 10) -> list[dict[str, str]]:
    """
    Return the last `limit` messages in the simple {role, content} format
    expected by the RAG pipeline's memory module.
    """
    history = st.session_state.messages[-limit:]
    return [{"role": m["role"], "content": m["content"]} for m in history]


# ---------------------------------------------------------------------------
# Mood
# ---------------------------------------------------------------------------

def set_mood(mood: str | None) -> None:
    """
    Set the active mood.
    Appends to mood_history if it's a new mood (deduplicates consecutive
    identical moods so rapidly double-clicking doesn't spam the list).
    """
    if mood and mood != st.session_state.current_mood:
        st.session_state.mood_history.append(mood)
    st.session_state.current_mood = mood


def get_mood() -> str | None:
    """Return the currently active mood."""
    return st.session_state.current_mood


def clear_mood() -> None:
    """Deselect the current mood."""
    st.session_state.current_mood = None


# ---------------------------------------------------------------------------
# Filters (dietary + cuisine)
# ---------------------------------------------------------------------------

def set_dietary_filters(filters: list[str]) -> None:
    st.session_state.dietary_filters = filters


def set_cuisine_filters(filters: list[str]) -> None:
    st.session_state.cuisine_filters = filters


def get_filters() -> dict[str, list[str]]:
    return {
        "dietary": st.session_state.dietary_filters,
        "cuisine": st.session_state.cuisine_filters,
    }


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def set_pipeline(pipeline: Any) -> None:
    """Store the initialised RAG pipeline (avoids reinitialising on every rerun)."""
    st.session_state.rag_pipeline = pipeline


def get_pipeline() -> Any | None:
    return st.session_state.rag_pipeline


def has_pipeline() -> bool:
    return st.session_state.rag_pipeline is not None


# ---------------------------------------------------------------------------
# Loading state
# ---------------------------------------------------------------------------

def set_loading(loading: bool) -> None:
    st.session_state.is_loading = loading


def is_loading() -> bool:
    return st.session_state.is_loading


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def get_latest_recommendations() -> list[dict]:
    return st.session_state.recommendations


def clear_recommendations() -> None:
    st.session_state.recommendations = []


# ---------------------------------------------------------------------------
# Sidebar metrics helpers
# ---------------------------------------------------------------------------

def get_session_stats() -> dict[str, Any]:
    """Return a snapshot of session statistics for the sidebar."""
    return {
        "session_id":    st.session_state.session_id[:8],   # short display
        "message_count": st.session_state.message_count,
        "mood_history":  st.session_state.mood_history,
        "current_mood":  st.session_state.current_mood,
        "filters":       get_filters(),
    }