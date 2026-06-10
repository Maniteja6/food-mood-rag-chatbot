"""
rag/prompt_builder.py
═════════════════════
Constructs the full LLM message array for each query turn.

The PromptBuilder class owns everything related to turning retrieved food
documents + mood context + conversation history into a prompt the LLM can
act on.  It has zero knowledge of embeddings, vector stores, or LLM clients
— it only builds strings and message dicts.

Message structure produced
──────────────────────────
    [
        {"role": "system",    "content": <system_prompt>},
        {"role": "user",      "content": <previous user turn>},   # from history
        {"role": "assistant", "content": <previous reply>},        # from history
        ...
        {"role": "user",      "content": <current query>},
    ]

System prompt sections (in order)
──────────────────────────────────
    1. Persona          Who MoodBite is and how it behaves
    2. Format rules     Exact output format: prose + ```json``` block
    3. Mood context     Mood-specific guidance from config/moods.py
    4. Food context     Numbered list of retrieved dishes with metadata

Public API
──────────
    PromptBuilder(config)
    builder.build(query, mood, retrieved, history)  → list[Message]
    builder.format_context(retrieved)               → str   (for debug)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Type aliases — kept local so this module has no rag.* circular imports
Message      = dict[str, str]
RetrievedDoc = dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ─────────────────────────────────────────────────────────────────────────────

_PERSONA = (
    "You are MoodBite — a warm, knowledgeable, and enthusiastic food "
    "recommendation assistant. Your entire purpose is to suggest dishes "
    "that perfectly match how the user is feeling right now.\n\n"
    "Your personality:\n"
    "  - Warm and genuinely caring, not robotic or clinical\n"
    "  - Culturally curious and respectful of all cuisines\n"
    "  - Specific: you name real dishes, not vague categories\n"
    "  - Honest: you always explain WHY a dish suits the user's mood\n"
    "  - Attentive: you never recommend anything that violates stated "
    "dietary restrictions\n"
    "  - Concise: prose responses are 2–4 sentences, never a wall of text"
)

_FORMAT_INSTRUCTIONS = (
    "OUTPUT FORMAT — follow this exactly every time:\n\n"
    "PART 1 — PROSE RESPONSE (required):\n"
    "  Write 2–4 warm, conversational sentences.\n"
    "  • Acknowledge the mood if one is provided\n"
    "  • Name 2–3 dishes naturally in the prose\n"
    "  • Give a one-line reason why each suits the mood\n"
    "  • End with a gentle question (e.g. 'Does that sound good?')\n\n"
    "PART 2 — FOOD CARDS JSON (required):\n"
    "  Immediately after the prose, output a fenced JSON block:\n\n"
    "  ```json\n"
    "  [\n"
    "    {\n"
    '      "name":         "Dish Name",\n'
    '      "cuisine":      "Cuisine Type",\n'
    '      "description":  "1–2 sentence enticing description.",\n'
    '      "tags":         ["Tag1", "Tag2", "Tag3"],\n'
    '      "score":        0.92,\n'
    '      "emoji":        "🍜",\n'
    '      "prep_time":    "25 min",\n'
    '      "why_for_mood": "One sentence why this suits the mood."\n'
    "    }\n"
    "  ]\n"
    "  ```\n\n"
    "  Rules for the JSON block:\n"
    "  • Include 3–5 dishes\n"
    "  • Only recommend dishes from the FOOD CONTEXT section below\n"
    "  • Do not invent dishes not present in the context\n"
    "  • score must be a float 0.0–1.0 reflecting relevance\n"
    "  • tags: 2–4 short strings (e.g. 'Vegetarian', 'Comfort food', 'Quick')\n"
    "  • emoji: a single food-related emoji character\n"
    "  • why_for_mood: must specifically reference the user's mood"
)

# Fallback when no mood is active
_NO_MOOD_HINT = (
    "No specific mood has been set. Recommend delicious, well-rounded food "
    "based on what the user describes. Be enthusiastic and specific."
)


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs LLM message arrays for the MoodBite RAG pipeline.

    Parameters
    ──────────
    config
        Application Settings object.  PromptBuilder reads:
            cfg.conversation_memory_limit   int   max history turns to include
    """

    def __init__(self, config: Any) -> None:
        self._cfg = config
        logger.debug(
            f"PromptBuilder ready "
            f"(memory_limit={config.conversation_memory_limit})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        query:     str,
        mood:      Optional[str],
        retrieved: list[RetrievedDoc],
        history:   list[Message],
    ) -> list[Message]:
        """
        Build the complete message array for one LLM call.

        Args:
            query:     Raw user message text.
            mood:      Active mood key e.g. "cozy", or None.
            retrieved: List of RetrievedDoc dicts from the retriever.
            history:   Recent conversation turns [{role, content}, ...].
                       Trimmed to cfg.conversation_memory_limit pairs.

        Returns:
            List of {role, content} dicts ready to pass to any LLM provider.
            Structure: [system, ...history, user_turn]
        """
        system_prompt = self._build_system_prompt(mood, retrieved)
        user_content  = self._build_user_message(query, mood)
        trimmed_hist  = self._trim_history(history)

        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *trimmed_hist,
            {"role": "user",   "content": user_content},
        ]

        logger.debug(
            f"PromptBuilder.build() — "
            f"system={len(system_prompt)} chars  "
            f"history={len(trimmed_hist)} turns  "
            f"user='{user_content[:50]}…'"
        )
        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # System prompt assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        mood:      Optional[str],
        retrieved: list[RetrievedDoc],
    ) -> str:
        """
        Assemble the system prompt from four sections.
        Sections are joined with double newlines; empty sections are skipped.
        """
        sections = [
            _PERSONA,
            _FORMAT_INSTRUCTIONS,
            self._build_mood_section(mood),
            self.format_context(retrieved),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_mood_section(self, mood: Optional[str]) -> str:
        """
        Build the mood guidance block injected into the system prompt.

        Pulls:
            - prompt_hint     one-sentence guidance from MoodConfig
            - avoid list      things to steer away from
            - food_categories high-level food type hints
            - top_cuisines    cuisine suggestions

        Returns an empty string if mood is None or unknown.
        """
        if not mood:
            return f"MOOD CONTEXT:\n{_NO_MOOD_HINT}"

        from config.moods import get_mood, get_mood_prompt_context
        mood_cfg  = get_mood(mood)
        hint      = get_mood_prompt_context(mood)

        if mood_cfg is None:
            return (
                f"MOOD CONTEXT:\n"
                f"Current mood: {mood}\n"
                f"{hint}"
            )

        avoid_str      = ", ".join(mood_cfg.avoid)      if mood_cfg.avoid      else "none"
        food_cats_str  = ", ".join(mood_cfg.food_categories[:6])
        cuisines_str   = ", ".join(mood_cfg.top_cuisines[:5])

        lines = [
            f"MOOD CONTEXT — current mood: {mood_cfg.label.upper()} {mood_cfg.emoji}",
            f"",
            f"Guidance    : {hint}",
            f"Avoid       : {avoid_str}",
            f"Food types  : {food_cats_str}",
            f"Top cuisines: {cuisines_str}",
            f"Spice level : {mood_cfg.spice_preference}",
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Food context formatter (also public for debug UI)
    # ─────────────────────────────────────────────────────────────────────────

    def format_context(self, retrieved: list[RetrievedDoc]) -> str:
        """
        Format retrieved food documents as a numbered menu for the prompt.

        Each entry contains all fields the LLM needs to write a good card:
        name, cuisine, meal type, score, prep time, calories, moods, dietary
        tags, spice level, ingredients (truncated), and description (truncated).

        Args:
            retrieved: List of RetrievedDoc dicts from the Retriever.

        Returns:
            A multi-line string ready to inject into the system prompt.
            Returns a "no results" fallback string if retrieved is empty.
        """
        if not retrieved:
            return (
                "FOOD CONTEXT:\n"
                "No dishes were retrieved from the database for this query. "
                "Draw on your general culinary knowledge to make helpful "
                "suggestions, but tell the user you're recommending from "
                "general knowledge rather than the food database."
            )

        lines = [
            f"FOOD CONTEXT — {len(retrieved)} dishes retrieved "
            f"(recommend only from these):",
        ]

        for i, doc in enumerate(retrieved, start=1):
            meta = doc.get("metadata", {})

            name        = meta.get("name",           "Unknown dish")
            cuisine     = meta.get("cuisine",        "Unknown")
            meal_type   = meta.get("meal_type",      "")
            moods       = meta.get("moods",          "")
            dietary     = meta.get("dietary_tags",   "none specified")
            spice       = meta.get("spice_level",    "unknown")
            prep        = meta.get("prep_time_mins", "?")
            calories    = meta.get("calories_approx","?")
            servings    = meta.get("servings",       "?")
            method      = meta.get("cooking_method", "")
            flavour     = meta.get("flavour_profile","")
            texture     = meta.get("texture",        "")
            occasion    = meta.get("occasion",       "")
            ingredients = meta.get("ingredients",    "")[:150]

            # Prefer metadata description; fall back to raw document text
            description = (
                meta.get("description", "") or doc.get("document", "")
            )[:220]

            score = doc.get("score", 0.0)

            entry_lines = [
                f"",
                f"[{i}] {name}",
                f"    Cuisine    : {cuisine}  |  Meal type : {meal_type}",
                f"    Score      : {score:.2f}  |  Prep: {prep} min  |  ~{calories} kcal  |  Serves {servings}",
                f"    Mood tags  : {moods}",
                f"    Dietary    : {dietary}  |  Spice: {spice}",
            ]

            if method or flavour or texture:
                chars = "  |  ".join(
                    filter(None, [
                        f"Method: {method}" if method else "",
                        f"Flavour: {flavour}" if flavour else "",
                        f"Texture: {texture}" if texture else "",
                    ])
                )
                entry_lines.append(f"    Char.      : {chars}")

            if occasion:
                entry_lines.append(f"    Occasion   : {occasion}")

            if ingredients:
                entry_lines.append(f"    Ingredients: {ingredients}")

            if description:
                entry_lines.append(f"    Description: {description}")

            lines.extend(entry_lines)

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # User message builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_user_message(self, query: str, mood: Optional[str]) -> str:
        """
        Build the user-turn content string.

        Prepends the mood so the LLM has it in the direct conversational
        context as well as in the system prompt.
        """
        query = query.strip()
        if mood:
            from config.moods import get_mood
            mood_cfg = get_mood(mood)
            label    = mood_cfg.label if mood_cfg else mood.capitalize()
            emoji    = mood_cfg.emoji if mood_cfg else ""
            return f"I'm feeling {label} {emoji}. {query}"
        return query

    # ─────────────────────────────────────────────────────────────────────────
    # History trimmer
    # ─────────────────────────────────────────────────────────────────────────

    def _trim_history(self, history: list[Message]) -> list[Message]:
        """
        Trim the conversation history to fit within the memory limit.

        The limit is interpreted as number of full turns (user + assistant),
        so we keep the last ``limit * 2`` individual messages.

        We also ensure the history slice always starts on a user message
        (not an assistant message) to avoid confusing the LLM with an
        orphaned assistant turn at the start of the context.

        Args:
            history: Full session history as [{"role", "content"}, ...].

        Returns:
            Trimmed history slice.
        """
        if not history:
            return []

        limit   = self._cfg.conversation_memory_limit
        # Keep last limit*2 messages (limit full turns)
        trimmed = history[-(limit * 2):]

        # Ensure we don't start with an assistant message
        if trimmed and trimmed[0].get("role") == "assistant":
            trimmed = trimmed[1:]

        return trimmed