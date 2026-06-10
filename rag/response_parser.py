"""
rag/response_parser.py
══════════════════════
Parses raw LLM response strings into structured output for the MoodBite UI.

The LLM is instructed (via PromptBuilder) to output:

    <2–4 sentence prose response>

    ```json
    [
      {
        "name": "...", "cuisine": "...", "description": "...",
        "tags": [...], "score": 0.9, "emoji": "🍜",
        "prep_time": "25 min", "why_for_mood": "..."
      },
      ...
    ]
    ```

ResponseParser splits this into two parts and handles every failure mode:
    1. JSON block found and valid     → prose + parsed cards
    2. JSON block found but malformed → prose + synthesised cards from metadata
    3. No JSON block at all           → full text as prose + synthesised cards
    4. LLM returned empty string      → fallback prose + synthesised cards

Synthesis fallback
──────────────────
When JSON parsing fails, ``synthesise_cards_from_retrieved()`` builds food
card dicts directly from the vector store metadata that was retrieved for
this query.  This guarantees the UI always has food cards to render, even
when the LLM misbehaves.

Public API
──────────
    ResponseParser()
    parser.parse(raw, retrieved)        → ParsedResponse
    parser.synthesise_cards(retrieved)  → list[FoodCard]

    ParsedResponse  dataclass:
        prose           str
        food_cards      list[FoodCard]
        json_found      bool    True if a ```json``` block was extracted
        json_valid      bool    True if the block parsed without errors
        synthesis_used  bool    True if cards came from fallback synthesis
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Type aliases
RetrievedDoc = dict[str, Any]
FoodCard     = dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedResponse:
    """
    The structured output of one ResponseParser.parse() call.

    Attributes
    ──────────
    prose
        Cleaned conversational text for the chat bubble.
        Never empty — a fallback is generated if the LLM gave nothing useful.
    food_cards
        List of food card dicts for the UI card grid.
        Always has at least 1 entry (synthesised from metadata if needed).
    json_found
        True if a ```json...``` block was found in the raw LLM response.
    json_valid
        True if the JSON block parsed successfully and yielded a list.
    synthesis_used
        True if food_cards were synthesised from retrieved metadata rather
        than parsed from the LLM response.
    raw_response
        The original unmodified LLM output string (for debug logging).
    """
    prose:          str
    food_cards:     list[FoodCard]    = field(default_factory=list)
    json_found:     bool              = False
    json_valid:     bool              = False
    synthesis_used: bool              = False
    raw_response:   str               = ""

    def to_dict(self) -> dict:
        return {
            "prose":          self.prose,
            "food_cards":     self.food_cards,
            "json_found":     self.json_found,
            "json_valid":     self.json_valid,
            "synthesis_used": self.synthesis_used,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ResponseParser
# ─────────────────────────────────────────────────────────────────────────────

class ResponseParser:
    """
    Parses raw LLM output into a ParsedResponse.

    Stateless — safe to create once and reuse across many parse() calls.
    """

    # Regex that matches the ```json ... ``` fenced block
    _JSON_FENCE_RE = re.compile(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        re.IGNORECASE,
    )

    # Regex that matches stray ``` markers left after extraction
    _STRAY_FENCE_RE = re.compile(r"```[a-z]*|```", re.IGNORECASE)

    # ─────────────────────────────────────────────────────────────────────────
    # Main parse method
    # ─────────────────────────────────────────────────────────────────────────

    def parse(
        self,
        raw_response:  str,
        retrieved:     list[RetrievedDoc],
        mood:          str | None = None,
    ) -> ParsedResponse:
        """
        Parse a raw LLM response string into prose + food cards.

        Args:
            raw_response: The raw string from the LLM (may be empty).
            retrieved:    RetrievedDoc list from the Retriever.
                          Used as fallback source for food cards.
            mood:         Active mood key (used in fallback prose).

        Returns:
            ParsedResponse dataclass.
        """
        raw_response = raw_response or ""
        result       = ParsedResponse(
            prose=raw_response.strip(),
            raw_response=raw_response,
        )

        if not raw_response.strip():
            # LLM returned nothing — full fallback
            result.prose          = self._empty_fallback_prose(mood)
            result.food_cards     = self.synthesise_cards(retrieved)
            result.synthesis_used = True
            logger.warning("LLM returned empty response — using full fallback.")
            return result

        # ── Try to extract and parse ```json``` block ─────────────────────────
        json_match = self._JSON_FENCE_RE.search(raw_response)

        if json_match:
            result.json_found = True
            json_str = json_match.group(1).strip()

            food_cards, parse_ok = self._parse_json_block(json_str)
            result.json_valid = parse_ok

            if parse_ok and food_cards:
                result.food_cards = food_cards
                # Prose is everything before the JSON fence
                prose_raw      = raw_response[: json_match.start()].strip()
                result.prose   = self._sanitise_prose(prose_raw)

                # If the LLM put nothing before the fence, generate prose
                if not result.prose:
                    result.prose = self._synthesise_prose(food_cards, mood)

            else:
                # JSON found but malformed — use full text as prose + synthesise
                result.prose          = self._sanitise_prose(raw_response)
                result.food_cards     = self.synthesise_cards(retrieved)
                result.synthesis_used = True
                logger.warning(
                    "JSON block found but could not be parsed — "
                    "synthesising cards from retrieved metadata."
                )
        else:
            # No JSON fence at all — treat entire response as prose + synthesise
            result.prose          = self._sanitise_prose(raw_response)
            result.food_cards     = self.synthesise_cards(retrieved)
            result.synthesis_used = True
            logger.debug(
                "No ```json``` block in LLM response — "
                "synthesising cards from retrieved metadata."
            )

        # ── Safety net: always have at least 1 card ───────────────────────────
        if not result.food_cards and retrieved:
            result.food_cards     = self.synthesise_cards(retrieved)
            result.synthesis_used = True

        logger.debug(
            f"parse() → prose={len(result.prose)} chars  "
            f"cards={len(result.food_cards)}  "
            f"json_found={result.json_found}  "
            f"json_valid={result.json_valid}  "
            f"synthesis={result.synthesis_used}"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # JSON parsing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_json_block(
        self,
        json_str: str,
    ) -> tuple[list[FoodCard], bool]:
        """
        Attempt to parse a JSON string into a list of food card dicts.

        Tries two strategies:
            1. Standard json.loads()
            2. If that fails, a lenient repair that closes unclosed brackets
               (handles LLM truncation mid-stream)

        Returns:
            (cards, success)
            cards    list[FoodCard]   normalised card dicts; empty on failure
            success  bool             True if parsing succeeded
        """
        # Strategy 1: standard parse
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                cards = [self._normalise_card(c) for c in data if isinstance(c, dict)]
                return cards, True
            elif isinstance(data, dict):
                # LLM sometimes wraps the list: {"recommendations": [...]}
                for key in ("recommendations", "dishes", "results", "food"):
                    if key in data and isinstance(data[key], list):
                        cards = [self._normalise_card(c) for c in data[key]]
                        return cards, True
            logger.warning(f"JSON parsed but unexpected type: {type(data)}")
            return [], False

        except json.JSONDecodeError:
            pass

        # Strategy 2: lenient repair — try to close truncated JSON
        repaired = self._repair_json(json_str)
        if repaired:
            try:
                data = json.loads(repaired)
                if isinstance(data, list):
                    cards = [self._normalise_card(c) for c in data if isinstance(c, dict)]
                    if cards:
                        logger.debug("Recovered cards via JSON repair.")
                        return cards, True
            except json.JSONDecodeError:
                pass

        return [], False

    def _repair_json(self, json_str: str) -> str | None:
        """
        Attempt a simple repair for truncated JSON arrays.

        LLMs occasionally get cut off mid-stream, leaving something like:
            [{"name": "Ramen", "cuisine": "Japanese", "descr
        We try to close any open string, then close the object, then the array.

        Returns the repaired string, or None if repair seems pointless.
        """
        s = json_str.strip()
        if not s.startswith("["):
            return None

        # Count unclosed structures
        open_braces  = s.count("{") - s.count("}")
        open_brackets = s.count("[") - s.count("]")

        # If the last character is in the middle of a string value, truncate
        # back to the last complete key-value pair
        last_complete = max(
            s.rfind('",'),
            s.rfind('"}\n'),
            s.rfind('"}'),
        )
        if last_complete > 0 and open_braces > 0:
            s = s[: last_complete + 2]   # include the closing quote + comma

        # Close all open braces and brackets
        s += "}" * max(0, open_braces)
        s += "]" * max(0, open_brackets)

        return s if len(s) > 10 else None

    # ─────────────────────────────────────────────────────────────────────────
    # Food card normalisation
    # ─────────────────────────────────────────────────────────────────────────

    def _normalise_card(self, raw: dict) -> FoodCard:
        """
        Ensure a food card dict has all expected keys with safe defaults.

        Handles missing keys, wrong types, and out-of-range scores.
        """
        # Name
        name = str(raw.get("name", "")).strip() or "Unknown Dish"

        # Cuisine
        cuisine = str(raw.get("cuisine", "")).strip()

        # Description — cap at 300 chars
        description = str(raw.get("description", "")).strip()[:300]

        # Tags — must be list of strings
        raw_tags = raw.get("tags", [])
        if isinstance(raw_tags, list):
            tags = [str(t).strip() for t in raw_tags if str(t).strip()][:5]
        elif isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()][:5]
        else:
            tags = []

        # Score — float 0–1
        try:
            score = float(raw.get("score", 0.5))
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.5

        # Emoji — single character or short string
        emoji = str(raw.get("emoji", "🍽️")).strip()
        if not emoji:
            emoji = _cuisine_to_emoji(cuisine)

        # Prep time
        prep_time = str(raw.get("prep_time", "")).strip()

        # Why for mood
        why_for_mood = str(raw.get("why_for_mood", "")).strip()[:200]

        return FoodCard(
            name=name,
            cuisine=cuisine,
            description=description,
            tags=tags,
            score=round(score, 3),
            emoji=emoji,
            prep_time=prep_time,
            why_for_mood=why_for_mood,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Synthesis fallback
    # ─────────────────────────────────────────────────────────────────────────

    def synthesise_cards(
        self,
        retrieved: list[RetrievedDoc],
    ) -> list[FoodCard]:
        """
        Build food card dicts directly from retrieved vector store metadata.

        Used when JSON parsing fails or the LLM omits the JSON block.
        Always returns at least a sensible empty list rather than crashing.

        Args:
            retrieved: List of RetrievedDoc dicts from the Retriever.

        Returns:
            List of FoodCard dicts (one per retrieved doc).
        """
        cards: list[FoodCard] = []

        for doc in retrieved:
            meta    = doc.get("metadata", {})
            cuisine = meta.get("cuisine", "")

            # Build tags from dietary + meal type
            raw_dietary = meta.get("dietary_tags", "")
            dietary_tags = (
                [t.strip() for t in raw_dietary.split(",") if t.strip()]
                if raw_dietary
                else []
            )
            meal_type = meta.get("meal_type", "")
            spice     = meta.get("spice_level", "")

            tags = dietary_tags[:2]
            if meal_type:
                tags.append(meal_type)
            if spice and spice not in ("None", "Unknown", "unknown"):
                tags.append(spice)
            tags = tags[:4]

            # Prep time string
            prep_mins = meta.get("prep_time_mins", "")
            prep_time = f"{prep_mins} min" if prep_mins else ""

            # Description — prefer metadata field, fall back to document text
            description = (
                meta.get("description", "") or doc.get("document", "")
            )[:220].strip()

            cards.append(FoodCard(
                name=         meta.get("name",         "Unknown Dish"),
                cuisine=      cuisine,
                description=  description,
                tags=         tags,
                score=        round(max(0.0, min(1.0, float(doc.get("score", 0.0)))), 3),
                emoji=        _cuisine_to_emoji(cuisine),
                prep_time=    prep_time,
                why_for_mood= "",
            ))

        return cards

    # ─────────────────────────────────────────────────────────────────────────
    # Prose helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sanitise_prose(self, text: str) -> str:
        """
        Remove markdown artefacts from prose text.

        Cleans:
            - Stray ``` fences left after JSON extraction
            - Excess blank lines (3+ → 2)
            - Leading/trailing whitespace
        """
        text = self._STRAY_FENCE_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _synthesise_prose(
        self,
        cards: list[FoodCard],
        mood:  str | None,
    ) -> str:
        """
        Generate a brief fallback prose line when the LLM omitted prose
        but did provide a valid JSON block.
        """
        if not cards:
            return "Here are some food recommendations that might suit you."

        names    = [c["name"]    for c in cards[:3] if c.get("name")]
        cuisines = [c["cuisine"] for c in cards[:3] if c.get("cuisine")]
        name_str = ", ".join(names) if names else "these dishes"

        if mood:
            return (
                f"Based on how you're feeling, I think {name_str} would hit "
                f"the spot perfectly. Each one is picked to match your {mood} "
                f"mood — give them a look!"
            )
        return (
            f"Here are my top picks for you: {name_str}. "
            f"Each one brings something special to the table!"
        )

    def _empty_fallback_prose(self, mood: str | None) -> str:
        """Return a safe prose response when the LLM returned nothing."""
        if mood:
            return (
                f"I'm sorry, I had a little trouble coming up with suggestions "
                f"just now. You're feeling {mood} — try asking me again and I'll "
                f"find something perfect for you!"
            )
        return (
            "I had a small hiccup just then. Could you try asking again? "
            "I'd love to find the perfect dish for you!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cuisine emoji map
# ─────────────────────────────────────────────────────────────────────────────

_CUISINE_EMOJI: dict[str, str] = {
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
}


def _cuisine_to_emoji(cuisine: str) -> str:
    """Return a food emoji for the given cuisine string."""
    return _CUISINE_EMOJI.get(cuisine.lower().strip(), "🍽️")