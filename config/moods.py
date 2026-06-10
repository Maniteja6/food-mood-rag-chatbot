"""
config/moods.py
═══════════════
The single source of truth for every mood supported by MoodBite.

This module exposes:

    MOOD_REGISTRY   dict[str, MoodConfig]   full config for each mood
    MOOD_KEYS       list[str]               ordered list of mood keys
    ALL_MOODS       list[MoodConfig]        ordered list of MoodConfig objects

    get_mood(key)               → MoodConfig | None
    get_mood_descriptors(key)   → list[str]   (for RAG query expansion)
    get_mood_prompt_context(key)→ str         (for LLM prompt builder)
    get_mood_filter_hints(key)  → dict        (for vector store metadata filter)
    mood_keys_for_display()     → list[dict]  (for Streamlit UI)

MoodConfig dataclass
─────────────────────
Each mood has these fields:

    key               str          Machine-readable key  e.g. "happy"
    label             str          Display label         e.g. "Happy"
    emoji             str          UI emoji              e.g. "😄"
    colour            str          Hex background for UI pill
    descriptors       list[str]    Semantic keywords appended to the RAG query
    food_categories   list[str]    Food type hints for the LLM prompt
    flavours          list[str]    Preferred flavour profiles (from dataset stats)
    textures          list[str]    Preferred textures (from dataset stats)
    occasions         list[str]    Matching occasions (from dataset stats)
    top_cuisines      list[str]    Top cuisine recommendations
    meal_types        list[str]    Preferred meal types
    avoid             list[str]    Things to steer away from
    spice_preference  str          "mild" | "mild to medium" | "any" etc.
    prompt_hint       str          One-sentence LLM prompt context line

All mood data is grounded in actual food_dataset.csv co-occurrence statistics
and the mood_food_mapping.json file generated alongside the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# MoodConfig dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MoodConfig:
    """Immutable configuration for a single mood."""

    key:              str
    label:            str
    emoji:            str
    colour:           str                    # hex background colour for UI pill

    # RAG & retrieval
    descriptors:      tuple[str, ...]        # appended to query for embedding
    food_categories:  tuple[str, ...]        # high-level food type hints
    flavours:         tuple[str, ...]        # preferred flavour profiles
    textures:         tuple[str, ...]        # preferred textures
    occasions:        tuple[str, ...]        # matching social occasions
    top_cuisines:     tuple[str, ...]        # best-fit cuisine types
    meal_types:       tuple[str, ...]        # matching meal types

    # Prompt & filter
    avoid:            tuple[str, ...]        # negative signals for LLM
    spice_preference: str                    # human-readable spice level
    prompt_hint:      str                    # injected into LLM system prompt

    # ── Helpers ───────────────────────────────────────────────────────────────

    def descriptor_string(self) -> str:
        """Return descriptors as a comma-separated string for query expansion."""
        return ", ".join(self.descriptors)

    def food_category_string(self) -> str:
        """Return food categories as a readable list string."""
        return ", ".join(self.food_categories)

    def cuisine_string(self) -> str:
        return ", ".join(self.top_cuisines)

    def to_ui_dict(self) -> dict:
        """Return the minimal dict needed by ui_components.py."""
        return {
            "key":    self.key,
            "label":  self.label,
            "emoji":  self.emoji,
            "colour": self.colour,
        }

    def to_filter_dict(self) -> dict:
        """
        Return a dict of metadata filter hints for the vector store retriever.
        These can be passed as `where` filters in ChromaDB.
        """
        return {
            "flavours":        list(self.flavours),
            "textures":        list(self.textures),
            "occasions":       list(self.occasions),
            "top_cuisines":    list(self.top_cuisines),
            "meal_types":      list(self.meal_types),
            "spice_preference": self.spice_preference,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mood registry — 20 moods, fully specified
# ─────────────────────────────────────────────────────────────────────────────

MOOD_REGISTRY: dict[str, MoodConfig] = {

    "happy": MoodConfig(
        key="happy", label="Happy", emoji="😄", colour="#FFF8E7",
        descriptors=(
            "uplifting", "bright", "joyful", "light", "fresh", "colourful",
            "celebratory", "cheerful", "energising", "vibrant", "feel-good",
            "sunshine", "zesty", "playful", "lively",
        ),
        food_categories=(
            "light meals", "fresh salads", "fruit-based dishes",
            "colourful bowls", "celebratory food", "crowd-pleasers", "sharing plates",
        ),
        flavours=("fragrant", "creamy", "delicate", "tangy", "citrusy", "sweet"),
        textures=("airy", "light", "chunky", "fluffy", "crispy"),
        occasions=("family gathering", "potluck", "celebration feast", "Sunday brunch"),
        top_cuisines=("French", "Caribbean", "Korean", "Japanese", "Mediterranean"),
        meal_types=("Main Course", "Salad", "Dessert", "Snack"),
        avoid=("very heavy", "extremely rich", "dense stodgy"),
        spice_preference="mild to medium",
        prompt_hint=(
            "The user is feeling happy and upbeat. Recommend light, vibrant, "
            "celebratory food that matches their positive energy. Prioritise "
            "fresh, colourful, and crowd-pleasing dishes."
        ),
    ),

    "sad": MoodConfig(
        key="sad", label="Sad", emoji="😢", colour="#EEF4FF",
        descriptors=(
            "comforting", "warming", "soothing", "nostalgic", "gentle",
            "soul-food", "healing", "tender", "soft", "consoling",
            "hug-in-a-bowl", "homestyle", "familiar", "cosy", "nourishing",
        ),
        food_categories=(
            "comfort food", "soups", "stews", "warm bowls", "noodle dishes",
            "mac and cheese", "mashed potato", "warm desserts", "hot chocolate",
            "casseroles", "pasta", "porridge",
        ),
        flavours=("briny", "sweet", "nutty", "fragrant", "buttery", "savoury"),
        textures=("silky", "fluffy", "melt-in-your-mouth", "velvety", "airy"),
        occasions=("solo lunch", "comfort meal", "rainy day", "weeknight dinner"),
        top_cuisines=("American", "Korean", "Japanese", "British", "Italian"),
        meal_types=("Main Course", "Soup", "Dessert"),
        avoid=("spicy", "challenging", "raw"),
        spice_preference="mild",
        prompt_hint=(
            "The user is feeling sad and needs comfort. Recommend warm, soothing, "
            "familiar comfort food. Think hug-in-a-bowl dishes — soups, stews, "
            "creamy pasta, or nostalgic home-cooking. Be gentle and nurturing in tone."
        ),
    ),

    "stressed": MoodConfig(
        key="stressed", label="Stressed", emoji="😤", colour="#FFF0F0",
        descriptors=(
            "calming", "indulgent", "familiar", "reliable", "grounding",
            "stress-relief", "satisfying", "uncomplicated", "soothing",
            "comfort", "unwinding", "treat yourself", "restorative", "simple",
        ),
        food_categories=(
            "comfort food", "warm dishes", "carbohydrate-rich", "indulgent",
            "dark chocolate", "ice cream", "pasta", "bread", "cheese dishes", "hot soup",
        ),
        flavours=("delicate", "nutty", "creamy", "sweet", "smoky", "savoury"),
        textures=("crunchy", "light", "dense", "melt-in-your-mouth", "tender"),
        occasions=("comfort meal", "quick bite", "date night", "weeknight dinner"),
        top_cuisines=("Thai", "Japanese", "Turkish", "Italian", "Indian"),
        meal_types=("Main Course", "Snack", "Dessert"),
        avoid=("complicated", "challenging", "time-consuming"),
        spice_preference="mild to medium",
        prompt_hint=(
            "The user is stressed and needs to unwind. Recommend reliable comfort food "
            "that feels like a treat — nothing too complicated or demanding to eat. "
            "Grounding, warm, and satisfying choices work best."
        ),
    ),

    "tired": MoodConfig(
        key="tired", label="Tired", emoji="😴", colour="#F0EEF8",
        descriptors=(
            "energising", "quick", "nourishing", "reviving", "easy",
            "simple", "iron-rich", "protein-packed", "pick-me-up",
            "sustaining", "fuel", "refreshing", "quick prep", "effortless",
        ),
        food_categories=(
            "quick meals", "energy-boosting", "protein-rich",
            "easy one-pan dishes", "smoothie bowls", "eggs", "rice bowls",
            "simple pasta", "takeaway-style", "sandwiches",
        ),
        flavours=("sweet", "umami-rich", "citrusy", "herbaceous", "tangy", "bold"),
        textures=("crispy", "flaky", "fluffy", "chewy", "light"),
        occasions=("quick bite", "weeknight dinner", "meal prep", "solo lunch"),
        top_cuisines=("Greek", "Moroccan", "Chinese", "Korean", "Japanese"),
        meal_types=("Main Course", "Snack", "Breakfast", "Soup"),
        avoid=("slow-cooked", "lengthy prep", "complicated techniques"),
        spice_preference="any",
        prompt_hint=(
            "The user is tired and low on energy. Recommend quick, nourishing, "
            "energising food that requires minimal effort to prepare or eat. "
            "Protein-rich and sustaining dishes are ideal."
        ),
    ),

    "romantic": MoodConfig(
        key="romantic", label="Romantic", emoji="💕", colour="#FFF0F5",
        descriptors=(
            "elegant", "rich", "sensual", "indulgent", "intimate",
            "luxurious", "special", "impressive", "sophisticated", "refined",
            "date-night", "candlelit", "fine dining", "share-worthy", "beautiful",
        ),
        food_categories=(
            "fine dining", "pasta dishes", "seafood", "steak",
            "chocolate desserts", "wine-friendly food", "sharing platters",
            "stuffed dishes", "slow-cooked", "French cuisine",
        ),
        flavours=("sweet and sour", "umami-rich", "nutty", "fragrant", "buttery", "creamy"),
        textures=("light", "airy", "tender", "silky", "melt-in-your-mouth"),
        occasions=("date night", "romantic dinner", "celebration feast"),
        top_cuisines=("French", "Italian", "Moroccan", "Turkish", "Mediterranean"),
        meal_types=("Main Course", "Starter", "Dessert"),
        avoid=("messy street food", "fast food", "pungent odours"),
        spice_preference="mild",
        prompt_hint=(
            "The user is in a romantic mood. Recommend elegant, sensual, and "
            "impressive dishes perfect for a date night — think beautifully plated "
            "pasta, seafood, or indulgent desserts. Sophisticated and memorable."
        ),
    ),

    "excited": MoodConfig(
        key="excited", label="Excited", emoji="🎉", colour="#FFF3E8",
        descriptors=(
            "bold", "vibrant", "festive", "adventurous", "fun",
            "punchy", "colourful", "party food", "crowd-pleasing",
            "sharing", "big flavours", "celebratory", "lively", "energetic",
        ),
        food_categories=(
            "street food", "sharing platters", "party food", "fried food",
            "bbq", "loaded dishes", "dumplings", "tacos", "wings",
            "colourful bowls", "festive food",
        ),
        flavours=("sweet and sour", "fragrant", "peppery", "umami-rich", "tangy", "spicy"),
        textures=("crispy", "velvety", "airy", "crunchy", "chewy"),
        occasions=("family gathering", "office party", "celebration feast", "potluck"),
        top_cuisines=("Brazilian", "Vietnamese", "Korean", "Mexican", "Thai"),
        meal_types=("Snack", "Main Course", "Starter"),
        avoid=("bland", "plain", "monotonous"),
        spice_preference="medium to spicy",
        prompt_hint=(
            "The user is excited and buzzing with energy. Recommend bold, vibrant, "
            "fun party food and sharing dishes with big flavours. Street food, "
            "loaded platters, and festive bites are perfect."
        ),
    ),

    "cozy": MoodConfig(
        key="cozy", label="Cozy", emoji="🍵", colour="#F5F0E8",
        descriptors=(
            "warming", "hearty", "comforting", "snug", "familiar",
            "homemade", "slow-cooked", "winter", "autumnal", "blanket food",
            "hot", "thick", "stodgy", "filling", "rustic",
        ),
        food_categories=(
            "stews", "casseroles", "hot soup", "roasts",
            "baked dishes", "gratins", "thick pasta", "warm curries",
            "bread pudding", "hot porridge", "hot chocolate",
        ),
        flavours=("savoury", "sweet", "briny", "bold", "smoky", "earthy"),
        textures=("sticky", "melt-in-your-mouth", "chunky", "dense", "hearty"),
        occasions=("rainy day", "comfort meal", "Sunday brunch", "weeknight dinner"),
        top_cuisines=("American", "Indian", "British", "French", "German"),
        meal_types=("Main Course", "Soup", "Dessert", "Breakfast"),
        avoid=("cold", "raw", "light salads"),
        spice_preference="mild",
        prompt_hint=(
            "The user wants something cozy. Think thick stews, hearty casseroles, "
            "baked gratins, and hot soups — the food equivalent of a warm blanket. "
            "Slow-cooked, rustic, and deeply satisfying."
        ),
    ),

    "adventurous": MoodConfig(
        key="adventurous", label="Adventurous", emoji="🌍", colour="#EDFAF4",
        descriptors=(
            "exotic", "bold", "unusual", "complex", "thrilling",
            "world flavours", "street food", "fermented", "rare", "discovery",
            "unfamiliar", "global", "boundary-pushing", "fusion", "unique",
        ),
        food_categories=(
            "exotic cuisines", "street food", "fermented dishes",
            "unusual ingredients", "fusion food", "raw dishes",
            "offal", "regional specialities", "rare ingredients",
        ),
        flavours=("umami-rich", "creamy", "tangy", "sweet", "earthy", "briny"),
        textures=("airy", "chewy", "light", "sticky", "crispy"),
        occasions=("camping trip", "holiday feast", "beach day", "date night"),
        top_cuisines=("Ethiopian", "Peruvian", "Indonesian", "Moroccan", "Filipino"),
        meal_types=("Main Course", "Starter", "Snack"),
        avoid=("plain", "familiar", "boring"),
        spice_preference="medium to very spicy",
        prompt_hint=(
            "The user is feeling adventurous and wants to try something new. "
            "Recommend exotic, unusual, or rarely-tried dishes from global cuisines. "
            "Push boundaries — fermented, spiced, and unfamiliar is exciting here."
        ),
    ),

    "anxious": MoodConfig(
        key="anxious", label="Anxious", emoji="😰", colour="#F0F4FF",
        descriptors=(
            "gentle", "mild", "calming", "soothing", "familiar",
            "safe", "plain", "easy-on-the-stomach", "light", "bland",
            "simple", "non-stimulating", "soft", "easy to eat", "comforting",
        ),
        food_categories=(
            "bland comfort food", "plain rice", "soup", "toast",
            "light pasta", "mashed potato", "porridge", "plain crackers",
            "chamomile tea", "light sandwiches", "oatmeal",
        ),
        flavours=("fragrant", "delicate", "savoury", "sweet", "buttery"),
        textures=("chewy", "dense", "flaky", "soft", "fluffy"),
        occasions=("solo lunch", "comfort meal", "weeknight dinner"),
        top_cuisines=("Japanese", "Mediterranean", "British", "Italian", "Greek"),
        meal_types=("Main Course", "Soup", "Breakfast"),
        avoid=("very spicy", "rich", "heavy", "caffeinated"),
        spice_preference="mild",
        prompt_hint=(
            "The user is feeling anxious. Recommend gentle, mild, easy-to-eat "
            "comfort food with no challenging flavours or textures. Simple, "
            "familiar, and easy on the stomach. Keep the tone calm and reassuring."
        ),
    ),

    "bored": MoodConfig(
        key="bored", label="Bored", emoji="😑", colour="#F5F5F0",
        descriptors=(
            "surprising", "interesting", "flavourful", "stimulating", "unique",
            "punchy", "new experience", "something different", "exciting flavours",
            "never tried", "fusion", "curiosity", "discovery", "bold", "complex",
        ),
        food_categories=(
            "new cuisines", "unusual dishes", "complex flavours",
            "street food", "tasting menus", "fusion", "interactive food",
            "build-your-own", "novelty food",
        ),
        flavours=("sweet and sour", "peppery", "fragrant", "spicy", "bold", "tangy"),
        textures=("dense", "chewy", "tender", "crispy", "crunchy"),
        occasions=("meal prep", "quick bite", "potluck", "date night"),
        top_cuisines=("Peruvian", "Vietnamese", "Spanish", "Korean", "Ethiopian"),
        meal_types=("Main Course", "Snack", "Starter"),
        avoid=("plain", "repetitive", "familiar"),
        spice_preference="medium to spicy",
        prompt_hint=(
            "The user is bored and needs something to excite their palate. "
            "Recommend surprising, flavourful, never-boring dishes — fusion food, "
            "unusual combinations, or cuisines they haven't tried. Stimulate curiosity."
        ),
    ),

    "nostalgic": MoodConfig(
        key="nostalgic", label="Nostalgic", emoji="🕰️", colour="#FAF0E6",
        descriptors=(
            "classic", "traditional", "homestyle", "reminiscent", "timeless",
            "grandma's recipe", "childhood favourite", "old school", "heritage",
            "retro", "home cooking", "memory", "vintage", "original", "authentic",
        ),
        food_categories=(
            "traditional recipes", "childhood favourites", "home cooking",
            "classic comfort food", "heritage dishes", "old-fashioned puddings",
            "family recipes", "grandmother's cooking",
        ),
        flavours=("briny", "smoky", "sweet and sour", "bold", "savoury", "sweet"),
        textures=("velvety", "airy", "light", "chewy", "crispy"),
        occasions=("family gathering", "holiday feast", "Sunday brunch", "comfort meal"),
        top_cuisines=("British", "American", "French", "Italian", "Filipino"),
        meal_types=("Main Course", "Dessert", "Breakfast", "Soup"),
        avoid=("trendy", "fusion", "deconstructed"),
        spice_preference="mild",
        prompt_hint=(
            "The user is feeling nostalgic. Recommend classic, traditional dishes "
            "that evoke childhood memories and home cooking — timeless recipes "
            "passed down through generations. Warm, familiar, and authentic."
        ),
    ),

    "celebratory": MoodConfig(
        key="celebratory", label="Celebratory", emoji="🥂", colour="#FFF9E6",
        descriptors=(
            "festive", "indulgent", "showstopping", "luxurious", "special",
            "party", "impressive", "extravagant", "decadent", "treat",
            "sharing feast", "milestone", "birthday", "champagne food", "rich",
        ),
        food_categories=(
            "whole roasts", "lobster", "champagne pairings", "layered cakes",
            "sharing feasts", "dumplings", "fine seafood", "whole fish",
            "elaborate desserts", "canapés",
        ),
        flavours=("sweet and sour", "umami-rich", "bitter", "citrusy", "rich", "buttery"),
        textures=("crispy", "fluffy", "flaky", "silky", "melt-in-your-mouth"),
        occasions=("celebration feast", "holiday feast", "family gathering", "birthday"),
        top_cuisines=("French", "Chinese", "Italian", "Japanese", "Lebanese"),
        meal_types=("Main Course", "Starter", "Dessert"),
        avoid=("plain", "everyday", "quick"),
        spice_preference="any",
        prompt_hint=(
            "The user is celebrating! Recommend showstopping, indulgent, and "
            "impressive dishes worthy of a special occasion — elaborate feasts, "
            "luxurious ingredients, and memorable centrepiece dishes."
        ),
    ),

    "lonely": MoodConfig(
        key="lonely", label="Lonely", emoji="🧸", colour="#F0F0FA",
        descriptors=(
            "comforting", "warming", "simple", "honest", "nourishing",
            "solo meal", "for one", "easy", "self-care", "cosy",
            "hug food", "kind", "gentle", "filling", "satisfying",
        ),
        food_categories=(
            "solo portions", "single-serve bowls", "simple pasta",
            "warm soups", "ramen", "omelette", "toast", "warm curries",
            "noodle bowls", "comfort food for one",
        ),
        flavours=("tangy", "earthy", "peppery", "umami-rich", "savoury", "sweet"),
        textures=("airy", "velvety", "fluffy", "silky", "warm"),
        occasions=("solo lunch", "comfort meal", "weeknight dinner", "rainy day"),
        top_cuisines=("Japanese", "Greek", "Mediterranean", "Indian", "Thai"),
        meal_types=("Main Course", "Soup", "Snack"),
        avoid=("large sharing portions", "complex"),
        spice_preference="mild to medium",
        prompt_hint=(
            "The user is feeling lonely. Recommend comforting, self-care food "
            "sized for one person — warm ramen, a simple bowl of pasta, or a "
            "satisfying soup. Nurturing, honest, and uncomplicated."
        ),
    ),

    "energetic": MoodConfig(
        key="energetic", label="Energetic", emoji="⚡", colour="#FFFBE6",
        descriptors=(
            "fresh", "light", "vibrant", "protein-rich", "zingy",
            "fuel", "performance", "clean eating", "post-workout", "lean",
            "high-energy", "power food", "active", "athletic", "nutritious",
        ),
        food_categories=(
            "protein bowls", "grilled chicken", "salads", "smoothie bowls",
            "eggs", "lean meat", "high-protein dishes", "fresh fish",
            "grain bowls", "vegetable stir-fry",
        ),
        flavours=("creamy", "umami-rich", "tangy", "fragrant", "citrusy", "herbaceous"),
        textures=("fluffy", "light", "flaky", "crispy", "tender"),
        occasions=("post-workout meal", "Sunday brunch", "quick bite", "meal prep"),
        top_cuisines=("Japanese", "Mediterranean", "Ethiopian", "Thai", "Vietnamese"),
        meal_types=("Main Course", "Salad", "Breakfast", "Snack"),
        avoid=("heavy", "fried", "stodgy", "high-sugar desserts"),
        spice_preference="any",
        prompt_hint=(
            "The user is feeling energetic and wants fuel to match. Recommend "
            "fresh, protein-rich, and nutritionally balanced food — grain bowls, "
            "lean grilled dishes, and vibrant salads. Clean and performance-ready."
        ),
    ),

    "sluggish": MoodConfig(
        key="sluggish", label="Sluggish", emoji="🐢", colour="#EFF8F0",
        descriptors=(
            "energising", "spiced", "stimulating", "reviving", "bold",
            "kick", "caffeine", "awakening", "metabolism-boosting",
            "lively flavours", "invigorating", "punchy", "bright",
        ),
        food_categories=(
            "spiced dishes", "coffee-infused", "chilli", "ginger-forward",
            "citrus dishes", "light energising soups", "protein-rich",
            "nutrient-dense", "antioxidant-rich",
        ),
        flavours=("tangy", "creamy", "peppery", "delicate", "bold", "spicy"),
        textures=("silky", "airy", "fluffy", "light", "crispy"),
        occasions=("Sunday brunch", "post-workout meal", "quick bite", "rainy day"),
        top_cuisines=("Spanish", "Ethiopian", "French", "Indian", "Korean"),
        meal_types=("Breakfast", "Snack", "Main Course", "Drink"),
        avoid=("very heavy", "sedating", "high-sugar"),
        spice_preference="medium to spicy",
        prompt_hint=(
            "The user is feeling sluggish and needs a wake-up call. Recommend "
            "invigorating, spiced, metabolism-boosting dishes with punchy flavours "
            "— ginger-forward, chilli-spiked, or citrus-bright food to revive them."
        ),
    ),

    "focused": MoodConfig(
        key="focused", label="Focused", emoji="🎯", colour="#EDF5FF",
        descriptors=(
            "clean", "light", "brain-boosting", "sustaining", "simple",
            "omega-3", "antioxidant", "low-distraction", "precise", "clear",
            "brain food", "cognitive", "steady energy", "no-nonsense",
        ),
        food_categories=(
            "brain food", "omega-3 rich", "nuts and seeds",
            "oily fish", "blueberries", "whole grains", "avocado",
            "green vegetables", "eggs", "light salads",
        ),
        flavours=("briny", "fragrant", "citrusy", "sweet and sour", "delicate", "herbaceous"),
        textures=("sticky", "flaky", "crispy", "light", "chewy"),
        occasions=("meal prep", "solo lunch", "quick bite", "weeknight dinner"),
        top_cuisines=("Japanese", "Mediterranean", "British", "Mexican", "Vietnamese"),
        meal_types=("Main Course", "Salad", "Snack", "Breakfast"),
        avoid=("heavy", "high-sugar crash", "sluggish"),
        spice_preference="mild",
        prompt_hint=(
            "The user is in focus mode. Recommend brain-boosting, clean, and "
            "sustaining food that won't cause an energy crash — omega-3 rich fish, "
            "whole grains, light protein. No heavy, distracting, or sugar-spike food."
        ),
    ),

    "heartbroken": MoodConfig(
        key="heartbroken", label="Heartbroken", emoji="💔", colour="#FFF0F8",
        descriptors=(
            "comforting", "indulgent", "nostalgic", "gentle", "soothing",
            "healing", "chocolate", "ice cream", "soft", "warm",
            "self-care", "feel-better", "kind to yourself", "sweet relief",
        ),
        food_categories=(
            "chocolate desserts", "ice cream", "warm brownies",
            "comfort food", "ramen", "pasta", "mac and cheese",
            "warm pastries", "hot soup", "sweet treats",
        ),
        flavours=("tangy", "spicy", "fragrant", "sweet", "rich", "nutty"),
        textures=("airy", "crispy", "flaky", "silky", "melt-in-your-mouth"),
        occasions=("comfort meal", "solo lunch", "rainy day", "weeknight dinner"),
        top_cuisines=("American", "Italian", "Japanese", "Turkish", "Mexican"),
        meal_types=("Dessert", "Main Course", "Snack"),
        avoid=("complicated", "sharing for two", "romantic"),
        spice_preference="mild",
        prompt_hint=(
            "The user is heartbroken. Be kind and warm in tone. Recommend the "
            "most indulgent, comforting food — chocolate, ice cream, ramen, rich "
            "pasta. This is not the time for healthy eating. Pure comfort only."
        ),
    ),

    "proud": MoodConfig(
        key="proud", label="Proud", emoji="🏆", colour="#FFFBEC",
        descriptors=(
            "celebratory", "elaborate", "showstopping", "impressive", "rich",
            "achievement", "reward", "treat", "hard-earned", "milestone",
            "special occasion", "I deserve this", "luxurious", "crafted",
        ),
        food_categories=(
            "elaborate dishes", "show-off recipes", "fine dining",
            "hard-to-make pastry", "impressive roasts",
            "layered cakes", "slow-braised meat", "elegant seafood",
        ),
        flavours=("sweet", "bitter", "earthy", "sweet and sour", "umami-rich", "rich"),
        textures=("airy", "flaky", "crunchy", "silky", "melt-in-your-mouth"),
        occasions=("celebration feast", "romantic dinner", "family gathering"),
        top_cuisines=("French", "Middle Eastern", "Lebanese", "Spanish", "Japanese"),
        meal_types=("Main Course", "Dessert", "Starter"),
        avoid=("quick", "simple", "plain"),
        spice_preference="any",
        prompt_hint=(
            "The user is feeling proud and wants to reward themselves. Recommend "
            "elaborate, impressive, and luxurious dishes that feel like a hard-earned "
            "treat — fine dining experiences, showstopping desserts, or elevated classics."
        ),
    ),

    "nervous": MoodConfig(
        key="nervous", label="Nervous", emoji="😬", colour="#F5F5FA",
        descriptors=(
            "mild", "familiar", "simple", "calming", "light",
            "plain", "safe choice", "well-known", "non-adventurous",
            "easy", "predictable", "easy to eat", "non-threatening",
        ),
        food_categories=(
            "plain food", "familiar favourites", "easy sandwiches",
            "plain pasta", "rice dishes", "simple soups", "toast",
            "crackers", "mild curry", "omelette",
        ),
        flavours=("umami-rich", "creamy", "fragrant", "delicate", "savoury"),
        textures=("airy", "light", "crispy", "fluffy", "tender"),
        occasions=("solo lunch", "quick bite", "weeknight dinner"),
        top_cuisines=("Italian", "Japanese", "British", "Spanish", "Mexican"),
        meal_types=("Main Course", "Snack", "Soup"),
        avoid=("very spicy", "unusual textures", "challenging"),
        spice_preference="mild",
        prompt_hint=(
            "The user is nervous and needs something safe and familiar. Recommend "
            "simple, well-known, easy-to-eat food with no challenging flavours. "
            "Predictable and comforting — nothing that requires bravery to try."
        ),
    ),

    "content": MoodConfig(
        key="content", label="Content", emoji="😌", colour="#F0FAF0",
        descriptors=(
            "satisfying", "balanced", "wholesome", "pleasant", "harmonious",
            "just right", "well-rounded", "complete", "measured", "perfect",
            "everyday favourite", "nothing to prove", "good food", "steady",
        ),
        food_categories=(
            "balanced meals", "grain bowls", "well-rounded dishes",
            "everyday favourites", "seasonal ingredients",
            "simple but quality", "honest cooking",
        ),
        flavours=("sweet and sour", "buttery", "nutty", "creamy", "savoury", "umami-rich"),
        textures=("melt-in-your-mouth", "dense", "tender", "chewy", "hearty"),
        occasions=("weeknight dinner", "family gathering", "date night", "solo lunch"),
        top_cuisines=("Thai", "Indian", "Chinese", "Italian", "Mediterranean"),
        meal_types=("Main Course", "Salad", "Side Dish"),
        avoid=("extreme", "overly complex"),
        spice_preference="any",
        prompt_hint=(
            "The user is feeling content and at ease. Recommend well-balanced, "
            "satisfying everyday food that's high quality without being over-the-top. "
            "Wholesome, harmonious, and quietly excellent."
        ),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Ordered lists (preserve insertion order = logical display order)
# ─────────────────────────────────────────────────────────────────────────────

MOOD_KEYS: list[str] = list(MOOD_REGISTRY.keys())
ALL_MOODS: list[MoodConfig] = list(MOOD_REGISTRY.values())


# ─────────────────────────────────────────────────────────────────────────────
# Public accessor functions
# ─────────────────────────────────────────────────────────────────────────────

def get_mood(key: str) -> Optional[MoodConfig]:
    """
    Return the MoodConfig for a given mood key, or None if not found.

    Args:
        key: Mood key string, e.g. "happy", "stressed". Case-insensitive.

    Returns:
        MoodConfig or None.
    """
    return MOOD_REGISTRY.get(key.lower().strip())


def get_mood_descriptors(key: str) -> list[str]:
    """
    Return the descriptor list for a mood — used to expand the RAG query.

    Returns an empty list if the mood is unknown (safe to use in join()).
    """
    mood = get_mood(key)
    return list(mood.descriptors) if mood else []


def get_mood_prompt_context(key: str) -> str:
    """
    Return the full LLM prompt hint for a mood.
    Injected into the system prompt by rag/prompt_builder.py.

    Returns a generic fallback string if the mood is unknown.
    """
    mood = get_mood(key)
    if mood:
        return mood.prompt_hint
    return (
        "Recommend delicious food that matches what the user is looking for. "
        "Be helpful, specific, and enthusiastic."
    )


def get_mood_filter_hints(key: str) -> dict:
    """
    Return metadata filter hints for the vector store retriever.
    Used by rag/retriever.py to optionally narrow results.

    Returns an empty dict if the mood is unknown.
    """
    mood = get_mood(key)
    return mood.to_filter_dict() if mood else {}


def mood_keys_for_display() -> list[dict]:
    """
    Return a list of minimal dicts for rendering the mood selector UI.
    Each dict: {"key", "label", "emoji", "colour"}

    Ordered as defined in MOOD_REGISTRY (the natural display order).
    """
    return [m.to_ui_dict() for m in ALL_MOODS]


def get_all_cuisine_names() -> list[str]:
    """Return a deduplicated sorted list of all cuisine names across all moods."""
    cuisines: set[str] = set()
    for mood in ALL_MOODS:
        cuisines.update(mood.top_cuisines)
    return sorted(cuisines)


def get_moods_for_cuisine(cuisine: str) -> list[str]:
    """Return all mood keys whose top_cuisines list includes the given cuisine."""
    return [
        m.key for m in ALL_MOODS
        if cuisine in m.top_cuisines
    ]


def build_expanded_query(base_query: str, mood_key: str) -> str:
    """
    Expand a user query with mood descriptors for richer semantic search.

    Args:
        base_query: The user's raw input, e.g. "something warm to eat"
        mood_key:   The active mood key, e.g. "cozy"

    Returns:
        An expanded query string, e.g.:
            "something warm to eat. Mood: cozy. warming hearty comforting snug..."
    """
    mood = get_mood(mood_key)
    if not mood:
        return base_query

    return (
        f"{base_query.rstrip('.')}. "
        f"Mood: {mood.label}. "
        f"{mood.descriptor_string()}. "
        f"Food types: {mood.food_category_string()}."
    )