"""
main.py — MoodBite Streamlit App Entrypoint

Run:
    streamlit run app/main.py

Flow:
    1. Inject CSS + initialise session state
    2. Render sidebar (filters, stats, clear-chat)
    3. Render header + mood selector
    4. Render chat history
    5. Accept chat input → call RAG pipeline → stream response → update state
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensures `from rag.pipeline import ...` works when the app is
# launched from the repo root via `streamlit run app/main.py`.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MoodBite — Food by Mood",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**MoodBite** — A mood-aware food recommendation chatbot "
            "powered by RAG and LLMs. No login required."
        ),
    },
)

# ---------------------------------------------------------------------------
# Internal imports (after sys.path is set)
# ---------------------------------------------------------------------------
from app.styles import inject_css
from app import session_state as ss
from app import ui_components as ui

# ---------------------------------------------------------------------------
# Lazy RAG pipeline import — we import lazily so the app loads even if the
# vector DB hasn't been built yet (it will show a warning instead of crashing).
# ---------------------------------------------------------------------------
_pipeline_import_error: str | None = None

try:
    from rag.pipeline import RAGPipeline
    _pipeline_available = True
except Exception as exc:                        # noqa: BLE001
    _pipeline_available = False
    _pipeline_import_error = str(exc)


# ===========================================================================
# Pipeline initialisation (cached across reruns via session_state)
# ===========================================================================

@st.cache_resource(show_spinner=False)
def _load_pipeline() -> "RAGPipeline | None":
    """
    Initialise the RAG pipeline once per Streamlit server process.
    @st.cache_resource caches the object across reruns and users.
    Returns None (with a logged warning) if initialisation fails.
    """
    if not _pipeline_available:
        return None
    try:
        pipeline = RAGPipeline()
        pipeline.initialise()
        return pipeline
    except Exception as exc:                    # noqa: BLE001
        st.warning(
            f"⚠️ RAG pipeline could not be loaded: {exc}. "
            "Make sure you've run `make ingest` to build the vector database."
        )
        return None


# ===========================================================================
# Main app
# ===========================================================================

def main() -> None:
    # ── 1. CSS & session ────────────────────────────────────────────────────
    inject_css()
    ss.init()

    # ── 2. Load pipeline into session state (once) ──────────────────────────
    if not ss.has_pipeline():
        pipeline = _load_pipeline()
        if pipeline:
            ss.set_pipeline(pipeline)

    # ── 3. Sidebar ──────────────────────────────────────────────────────────
    ui.render_sidebar()

    # ── 4. Main column layout ────────────────────────────────────────────────
    # We use a centred single column — wide layout gives breathing room on
    # large screens; the block-container max-width is capped in CSS.
    _, main_col, _ = st.columns([0.5, 9, 0.5])

    with main_col:
        # ── Header ──────────────────────────────────────────────────────────
        ui.render_header()

        # ── Pipeline warning ─────────────────────────────────────────────────
        if not ss.has_pipeline():
            if _pipeline_import_error:
                ui.render_error(
                    f"Could not import RAG pipeline: {_pipeline_import_error}. "
                    "Check your installation and run `make install`."
                )
            else:
                ui.render_pipeline_missing_warning()

        # ── Mood selector ────────────────────────────────────────────────────
        ui.render_mood_selector()

        # ── Chat divider ─────────────────────────────────────────────────────
        if ss.get_messages():
            st.markdown(
                '<div class="chat-divider">conversation</div>',
                unsafe_allow_html=True,
            )

        # ── Chat history ──────────────────────────────────────────────────────
        ui.render_chat_history()

        # ── Typing indicator (shown while loading) ───────────────────────────
        typing_placeholder = st.empty()
        if ss.is_loading():
            with typing_placeholder:
                ui.render_typing_indicator()

        # ── Chat input ────────────────────────────────────────────────────────
        _handle_chat_input(typing_placeholder)


# ===========================================================================
# Chat input handler
# ===========================================================================

def _handle_chat_input(typing_placeholder: "st.delta_generator.DeltaGenerator") -> None:
    """
    Render the chat input box and, when the user submits a message:
      1. Save the user message to session state
      2. Show typing indicator
      3. Call the RAG pipeline
      4. Save the assistant response + food cards to session state
      5. Rerun to refresh the UI
    """
    current_mood = ss.get_mood()
    placeholder_text = (
        f"Ask me anything — you're feeling {current_mood}..."
        if current_mood
        else "Describe how you feel or what you're craving..."
    )

    user_input = st.chat_input(placeholder_text, key="main_chat_input")

    if not user_input:
        return

    # Save user message
    ss.add_user_message(user_input)
    ss.set_loading(True)
    st.rerun()                          # rerun so typing indicator appears

    # NOTE: Everything below runs on the *next* rerun triggered above.
    # Streamlit re-executes main() from top; is_loading() == True causes
    # the typing indicator to appear, then we call the pipeline here.


def _run_pipeline_and_respond() -> None:
    """
    Called when is_loading() is True. Executes the RAG query and saves
    the result. Separated from _handle_chat_input to keep the flow clear.
    """
    pipeline = ss.get_pipeline()
    messages  = ss.get_messages()

    # The last message should be the user's (we just saved it)
    if not messages or messages[-1]["role"] != "user":
        ss.set_loading(False)
        return

    user_message = messages[-1]["content"]
    mood         = messages[-1].get("mood")
    filters      = ss.get_filters()
    history      = ss.get_conversation_for_rag(limit=10)

    try:
        if pipeline is None:
            response_text   = (
                "My food knowledge base isn't loaded yet. "
                "Run `make ingest` to build the vector database, then restart the app."
            )
            recommendations: list[dict] = []

        elif not pipeline.is_ready:
            # Pipeline exists but initialise() hasn't been called yet
            pipeline.initialise()
            result          = pipeline.query(
                query=user_message,
                mood=mood,
                filters=filters,
                history=history,
            )
            response_text   = result.get("response", "")
            recommendations = result.get("recommendations", [])

        else:
            result = pipeline.query(
                query=user_message,
                mood=mood,
                filters=filters,
                history=history,
            )
            response_text   = result.get("response", "")
            recommendations = result.get("recommendations", [])

            # Debug mode: show retrieval stats in the UI
            from config.settings import get_settings
            if get_settings().debug_mode:
                latency  = result.get("latency_ms", 0)
                n_docs   = len(result.get("retrieved_docs", []))
                st.caption(
                    f"🔍 Retrieved {n_docs} docs · "
                    f"Query: '{result.get('expanded_query','')[:60]}…' · "
                    f"{latency:.0f}ms"
                )

    except RuntimeError as exc:
        msg = str(exc)
        if "API key" in msg or "not set" in msg:
            response_text = (
                "⚠️ **API key not configured.** "
                "Please add your `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) "
                "to your `.env` file and restart the app. "
                "See `.env.example` for the full list of required variables."
            )
        else:
            response_text = (
                f"I ran into a configuration error: {msg}. "
                "Please check your `.env` file and restart the app."
            )
        recommendations = []
        traceback.print_exc()

    except Exception as exc:            # noqa: BLE001
        response_text   = (
            "I ran into a small hiccup fetching your recommendations. "
            "Please try again in a moment!"
        )
        recommendations = []
        traceback.print_exc()

    finally:
        ss.set_loading(False)

    ss.add_assistant_message(response_text, recommendations=recommendations)


# ===========================================================================
# Rerun interception for loading state
# ===========================================================================
# Streamlit re-runs the entire script on every interaction.
# When is_loading() is True we skip rendering and immediately call the
# pipeline so the round-trip is: user sends → rerun (shows typing) →
# pipeline runs → rerun (shows response).

def _intercept_loading() -> bool:
    """
    If the app is in loading state, run the pipeline and trigger a fresh
    rerun to display the result. Returns True if we intercepted.
    """
    if ss.is_loading():
        _run_pipeline_and_respond()
        st.rerun()
        return True
    return False


# ===========================================================================
# Entry
# ===========================================================================

if __name__ == "__main__" or True:
    # The `or True` ensures this block runs when Streamlit executes the file
    # as a module (it doesn't set __name__ == "__main__").
    ss.init()
    if not _intercept_loading():
        main()