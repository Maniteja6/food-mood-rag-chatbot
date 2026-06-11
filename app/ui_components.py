"""
ui_components.py — Reusable UI Components for MoodBite

All HTML/Markdown rendering and Streamlit widget groups live here.
main.py calls these functions; it never writes st.markdown() directly.

Components:
    render_header()               App logo + tagline
    render_mood_selector()        Horizontal mood pill buttons
    render_welcome_card()         Empty-state prompt card
    render_chat_history()         Full scrollable chat thread
    render_user_bubble()          Single user chat bubble
    render_assistant_bubble()     Single assistant bubble (+ food cards)
    render_food_cards()           Grid of food recommendation cards
    render_typing_indicator()     Animated three-dot indicator
    render_sidebar()              Filters, stats, settings
    render_clear_button()         Wipe-chat action button
"""

from __future__ import annotations

import streamlit as st

from app import session_state as ss

# ---------------------------------------------------------------------------
# Mood configuration
# ---------------------------------------------------------------------------

# Sourced from config/moods.py — all 20 moods in display order
def _load_moods() -> list[dict]:
    """Load mood display config from config/moods.py at import time."""
    try:
        from config.moods import ALL_MOODS
        return [
            {
                "key":   m.key,
                "label": m.label,
                "emoji": m.emoji,
                "css":   f"mood-{m.key}",
            }
            for m in ALL_MOODS
        ]
    except ImportError:
        # Fallback if config not on path yet (e.g. during testing)
        return [
            {"key": "happy",        "label": "Happy",        "emoji": "😄", "css": "mood-happy"},
            {"key": "sad",          "label": "Sad",          "emoji": "😢", "css": "mood-sad"},
            {"key": "stressed",     "label": "Stressed",     "emoji": "😤", "css": "mood-stressed"},
            {"key": "tired",        "label": "Tired",        "emoji": "😴", "css": "mood-tired"},
            {"key": "romantic",     "label": "Romantic",     "emoji": "💕", "css": "mood-romantic"},
            {"key": "excited",      "label": "Excited",      "emoji": "🎉", "css": "mood-excited"},
            {"key": "cozy",         "label": "Cozy",         "emoji": "🍵", "css": "mood-cozy"},
            {"key": "adventurous",  "label": "Adventurous",  "emoji": "🌍", "css": "mood-adventurous"},
            {"key": "anxious",      "label": "Anxious",      "emoji": "😰", "css": "mood-anxious"},
            {"key": "bored",        "label": "Bored",        "emoji": "😑", "css": "mood-bored"},
            {"key": "nostalgic",    "label": "Nostalgic",    "emoji": "🕰️", "css": "mood-nostalgic"},
            {"key": "celebratory",  "label": "Celebratory",  "emoji": "🥂", "css": "mood-celebratory"},
            {"key": "lonely",       "label": "Lonely",       "emoji": "🧸", "css": "mood-lonely"},
            {"key": "energetic",    "label": "Energetic",    "emoji": "⚡", "css": "mood-energetic"},
            {"key": "sluggish",     "label": "Sluggish",     "emoji": "🐢", "css": "mood-sluggish"},
            {"key": "focused",      "label": "Focused",      "emoji": "🎯", "css": "mood-focused"},
            {"key": "heartbroken",  "label": "Heartbroken",  "emoji": "💔", "css": "mood-heartbroken"},
            {"key": "proud",        "label": "Proud",        "emoji": "🏆", "css": "mood-proud"},
            {"key": "nervous",      "label": "Nervous",      "emoji": "😬", "css": "mood-nervous"},
            {"key": "content",      "label": "Content",      "emoji": "😌", "css": "mood-content"},
        ]

MOODS: list[dict] = _load_moods()

DIETARY_OPTIONS: list[str] = [
    "Vegetarian", "Vegan", "Gluten-Free",
    "Dairy-Free", "Nut-Free", "Halal", "Kosher",
]

CUISINE_OPTIONS: list[str] = [
    "Italian", "Japanese", "Indian", "Mexican", "Chinese",
    "Thai", "Mediterranean", "American", "French", "Korean",
    "Middle Eastern", "Spanish", "Greek", "Vietnamese", "Ethiopian",
    "Peruvian", "Turkish", "Moroccan", "Lebanese", "Brazilian",
    "British", "German", "Indonesian", "Filipino", "Caribbean",
    "Russian", "Polish", "Nigerian", "Argentinian", "Swedish",
]

# Food emoji map for cards — covers all 30 cuisines in the dataset
CUISINE_EMOJI: dict[str, str] = {
    "italian":       "🍝", "japanese":       "🍣", "indian":        "🍛",
    "mexican":       "🌮", "chinese":        "🥡", "thai":          "🍜",
    "mediterranean": "🥗", "american":       "🍔", "french":        "🥐",
    "korean":        "🍲", "middle eastern": "🧆", "spanish":       "🥘",
    "greek":         "🫒", "vietnamese":     "🍜", "ethiopian":     "🫓",
    "peruvian":      "🐟", "turkish":        "🥙", "moroccan":      "🫕",
    "lebanese":      "🧆", "brazilian":      "🥩", "british":       "🫖",
    "german":        "🥨", "indonesian":     "🍚", "filipino":      "🍖",
    "caribbean":     "🌴", "russian":        "🥟", "polish":        "🥣",
    "nigerian":      "🍲", "argentinian":    "🥩", "swedish":       "🫙",
    "default":       "🍽️",
}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Render the MoodBite logo and tagline."""
    st.markdown(
        """
        <div class="moodbite-header">
            <h1 class="moodbite-logo">Mood<span>Bite</span></h1>
            <p class="moodbite-tagline">Tell me how you feel. I'll tell you what to eat.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Mood selector
# ---------------------------------------------------------------------------

def render_mood_selector() -> None:
    """
    Render a row of mood pill buttons.
    Clicking a mood calls ss.set_mood() and triggers a rerun.
    The active mood pill is visually highlighted.
    """
    current_mood = ss.get_mood()

    st.markdown(
        '<p class="mood-section-title">How are you feeling right now?</p>',
        unsafe_allow_html=True,
    )

    # Render pills as Streamlit columns so each is a real clickable button.
    # We use 4 columns per row for a balanced look.
    cols_per_row = 4
    rows = [MOODS[i:i + cols_per_row] for i in range(0, len(MOODS), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, mood in zip(cols, row):
            with col:
                is_active = current_mood == mood["key"]
                label = f"{mood['emoji']} {mood['label']}"
                btn_type = "primary" if is_active else "secondary"

                if st.button(
                    label,
                    key=f"mood_btn_{mood['key']}",
                    use_container_width=True,
                    type=btn_type,          # type: ignore[arg-type]
                ):
                    if is_active:
                        # Clicking the active mood deselects it
                        ss.clear_mood()
                    else:
                        ss.set_mood(mood["key"])
                    st.rerun()

    # Inline hint when a mood is selected
    if current_mood:
        mood_label = next(
            (m["label"] for m in MOODS if m["key"] == current_mood), current_mood
        )
        st.markdown(
            f'<p style="text-align:center;font-size:0.82rem;color:var(--text-muted);'
            f'margin-top:0.25rem;">Mood set to <strong>{mood_label}</strong> — '
            f'now ask me anything or type what you\'re craving.</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Welcome / empty state card
# ---------------------------------------------------------------------------

def render_welcome_card() -> None:
    """Shown when the chat history is empty."""
    mood = ss.get_mood()
    if mood:
        mood_obj = next((m for m in MOODS if m["key"] == mood), None)
        emoji   = mood_obj["emoji"] if mood_obj else "🍽️"
        msg     = (
            f"You're feeling <strong>{mood_obj['label'].lower()}</strong> — "
            f"I've got the perfect dish for that. What would you like to know?"
        ) if mood_obj else "Tell me what you're craving!"
    else:
        emoji = "🍽️"
        msg   = (
            "Pick a mood above, or just tell me how you're feeling and "
            "I'll recommend something delicious that matches your vibe."
        )

    st.markdown(
        f"""
        <div class="welcome-card">
            <span class="welcome-emoji">{emoji}</span>
            <p>{msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

def render_chat_history() -> None:
    """
    Render the full conversation thread.
    Each message is rendered via its role-specific bubble function.
    """
    messages = ss.get_messages()
    if not messages:
        render_welcome_card()
        return

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in messages:
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        else:
            render_assistant_bubble(
                msg["content"],
                recommendations=msg.get("recommendations", []),
            )
    st.markdown('</div>', unsafe_allow_html=True)


def render_user_bubble(content: str) -> None:
    """Render a single user message bubble."""
    # Escape HTML to prevent injection
    safe_content = (
        content
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f"""
        <div class="chat-message user">
            <span class="chat-avatar">You</span>
            <div class="chat-bubble user-bubble">{safe_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_bubble(
    content: str,
    recommendations: list[dict] | None = None,
) -> None:
    """
    Render a single assistant message bubble.
    If `recommendations` is provided, renders food cards below the bubble.
    """
    st.markdown(
        f"""
        <div class="chat-message assistant">
            <span class="chat-avatar">MoodBite</span>
            <div class="chat-bubble assistant-bubble">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if recommendations:
        render_food_cards(recommendations)


# ---------------------------------------------------------------------------
# Typing indicator
# ---------------------------------------------------------------------------

def render_typing_indicator() -> None:
    """Show an animated three-dot typing indicator while the LLM is streaming."""
    st.markdown(
        """
        <div class="chat-message assistant">
            <span class="chat-avatar">MoodBite</span>
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Food recommendation cards
# ---------------------------------------------------------------------------

def render_food_cards(recommendations: list[dict]) -> None:
    """
    Render a responsive grid of food recommendation cards.

    Each dict in `recommendations` should have:
        name        str   Dish name
        cuisine     str   Cuisine type
        description str   Short description
        tags        list  Dietary / mood tags
        score       float (optional) Similarity score 0–1
        emoji       str   (optional) Override emoji
    """
    if not recommendations:
        return

    cards_html = '<div class="food-cards-grid">'

    for rec in recommendations:
        name      = rec.get("name", "Unknown Dish")
        cuisine   = rec.get("cuisine", "")
        desc      = rec.get("description", "")
        tags      = rec.get("tags", [])
        score     = rec.get("score")
        emoji     = rec.get("emoji") or _cuisine_emoji(cuisine)

        # Score badge
        score_html = ""
        if score is not None:
            pct = int(round(score * 100))
            score_html = f'<span class="food-card-score">✦ {pct}% match</span>'

        # Tag pills
        tag_pills = "".join(
            f'<span class="food-tag">{tag}</span>' for tag in tags[:4]
        )
        tags_html = f'<div class="food-card-tags">{tag_pills}</div>' if tag_pills else ""

        # Cuisine line
        cuisine_html = (
            f'<p class="food-card-cuisine">{cuisine.title()}</p>' if cuisine else ""
        )

        cards_html += f"""
        <div class="food-card">
            {score_html}
            <span class="food-card-emoji">{emoji}</span>
            <div class="food-card-body">
                <h3 class="food-card-name">{name}</h3>
                {cuisine_html}
                <p class="food-card-desc">{desc}</p>
                {tags_html}
            </div>
        </div>
        """

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def _cuisine_emoji(cuisine: str) -> str:
    """Return an emoji for a given cuisine string."""
    return CUISINE_EMOJI.get(cuisine.lower(), CUISINE_EMOJI["default"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """
    Render the left sidebar with:
      - Dietary & cuisine filters
      - Session stats (message count, mood history)
      - App settings (LLM model info)
      - Clear chat button
    """
    with st.sidebar:
        # ── Brand mark ──────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding: 1rem 0 0.5rem; text-align: center;">
                <h2 style="font-family: 'Playfair Display', serif;
                            font-size: 1.4rem; margin: 0;
                            color: #EDD28A !important;">
                    🍽️ MoodBite
                </h2>
                <p style="font-size: 0.75rem; color: rgba(250,246,240,0.5);
                           margin: 0.2rem 0 0; letter-spacing: 0.05em;">
                    Food · Mood · Magic
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Dietary filters ──────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.72rem;letter-spacing:0.12em;'
            'text-transform:uppercase;color:rgba(250,246,240,0.5);'
            'margin-bottom:0.4rem;">Dietary Preferences</p>',
            unsafe_allow_html=True,
        )
        selected_dietary = st.multiselect(
            label="dietary_filters_label",
            options=DIETARY_OPTIONS,
            default=ss.get_filters()["dietary"],
            label_visibility="collapsed",
            key="dietary_multiselect",
            placeholder="Any diet",
        )
        ss.set_dietary_filters(selected_dietary)

        # ── Cuisine filters ──────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.72rem;letter-spacing:0.12em;'
            'text-transform:uppercase;color:rgba(250,246,240,0.5);'
            'margin-top:0.8rem;margin-bottom:0.4rem;">Cuisine Style</p>',
            unsafe_allow_html=True,
        )
        selected_cuisine = st.multiselect(
            label="cuisine_filters_label",
            options=CUISINE_OPTIONS,
            default=ss.get_filters()["cuisine"],
            label_visibility="collapsed",
            key="cuisine_multiselect",
            placeholder="Any cuisine",
        )
        ss.set_cuisine_filters(selected_cuisine)

        st.markdown("---")

        # ── Session stats ────────────────────────────────────────────────────
        stats = ss.get_session_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats["message_count"])
        with col2:
            st.metric("Moods tried", len(stats["mood_history"]))

        if stats["mood_history"]:
            mood_display = " → ".join(
                next(
                    (m["emoji"] for m in MOODS if m["key"] == mk),
                    mk,
                )
                for mk in stats["mood_history"][-5:]     # last 5
            )
            st.markdown(
                f'<p style="font-size:0.78rem;color:rgba(250,246,240,0.5);'
                f'margin-top:0.25rem;">Journey: {mood_display}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Clear chat ───────────────────────────────────────────────────────
        render_clear_button()

        # ── Footer ───────────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.7rem;color:rgba(250,246,240,0.3);'
            'text-align:center;margin-top:2rem;line-height:1.6;">'
            'Powered by RAG + LLM<br>'
            '© 2024 MoodBite</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Clear chat button
# ---------------------------------------------------------------------------

def render_clear_button() -> None:
    """Render a button that wipes the chat history."""
    if st.button("🗑️  Clear Chat", use_container_width=True):
        ss.clear_messages()
        ss.clear_recommendations()
        st.rerun()


# ---------------------------------------------------------------------------
# Error / info banners
# ---------------------------------------------------------------------------

def render_error(message: str) -> None:
    """Render a styled error banner."""
    st.error(f"⚠️ {message}", icon=None)


def render_info(message: str) -> None:
    """Render a styled info banner."""
    st.info(f"ℹ️ {message}", icon=None)


def render_pipeline_missing_warning() -> None:
    """Warn the user if the RAG pipeline hasn't been initialised."""
    st.warning(
        "🔧 The RAG pipeline is not initialised. "
        "Run `make ingest` to build the vector database, "
        "then restart the app.",
        icon=None,
    )