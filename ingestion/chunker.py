"""
ingestion/chunker.py
════════════════════
Transforms a list of FoodRecord objects into a list of Chunk objects —
embeddable text units that will be stored in the vector database.

Why chunking matters for food data
────────────────────────────────────
Each FoodRecord has rich structured data (name, cuisine, moods, ingredients,
description, dietary tags, etc.). Simply embedding the raw description column
loses most of that signal. This chunker builds a carefully crafted "document
string" per record that:

  1. Packs ALL semantically useful fields into one coherent sentence so the
     embedding captures mood→food relationships strongly.
  2. Keeps the chunk self-contained — the retriever never needs to look up
     the original CSV row; everything needed to answer the user and render
     a food card lives in the Chunk.
  3. Stays under ~300 tokens so OpenAI / HuggingFace embeddings work
     without truncation (text-embedding-3-small supports 8191 tokens, but
     shorter focused chunks retrieve better in practice).

Chunk document format (what gets embedded)
───────────────────────────────────────────
  Dish: <name>
  Cuisine: <cuisine> | Meal type: <meal_type>
  Moods: <mood1>, <mood2>, ...
  Ingredients: <ingredient1>, <ingredient2>, ...
  Dietary: <tag1>, <tag2> | Spice: <spice_level>
  Cooking: <method> | Flavour: <profile> | Texture: <texture>
  Occasion: <occasion> | Prep: <N> min | ~<cal> kcal
  Description: <full description text>

Public API
──────────
    chunk_records(records, mood_mapping)  → list[Chunk]
    save_chunks(chunks, path)             → None
    load_chunks(path)                     → list[Chunk]
    Chunk                                 dataclass
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from ingestion.loader import FoodRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single unit ready to be embedded and stored in the vector database.

    Fields
    ──────
    chunk_id    Unique string ID (same as record id — one chunk per record).
    document    The full text string that gets passed to the embedding model.
    metadata    Flat dict of scalar fields stored alongside the vector.
                Used by the retriever for metadata filtering and for
                rendering food cards in the UI — no CSV lookup needed.
    """
    chunk_id:  str
    document:  str
    metadata:  dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            document=d["document"],
            metadata=d["metadata"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def chunk_records(
    records: list[FoodRecord],
    mood_mapping: dict[str, list[str]] | None = None,
) -> list[Chunk]:
    """
    Convert a list of FoodRecord objects into a list of Chunk objects.

    One Chunk is produced per FoodRecord. The chunk document is built by
    ``_build_document()``, which weaves all structured fields into a single
    dense text block that embeds well for mood-based semantic search.

    Args:
        records:      Validated FoodRecord list from loader.load_food_dataset().
        mood_mapping: Optional dict from loader.load_mood_mapping(). When
                      provided, mood descriptor keywords (e.g. "comforting",
                      "warming") are appended to the document for moods that
                      appear in the record, enriching the embedding signal.

    Returns:
        List of Chunk objects, same length as ``records`` (no records dropped).
    """
    if mood_mapping is None:
        mood_mapping = {}

    chunks: list[Chunk] = []

    for record in records:
        document = _build_document(record, mood_mapping)
        metadata = _build_metadata(record)
        chunks.append(Chunk(
            chunk_id=record.id,
            document=document,
            metadata=metadata,
        ))

    logger.info(f"Created {len(chunks):,} chunks from {len(records):,} records.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Document builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_document(
    record: FoodRecord,
    mood_mapping: dict[str, list[str]],
) -> str:
    """
    Build the embeddable document string for a single FoodRecord.

    Structure
    ─────────
    Structured fields come first (strong signal for keyword+semantic overlap),
    then the free-text description (rich contextual signal), then mood
    descriptors from the mapping (boosts retrieval for mood queries).

    All sections are newline-separated so the text is readable and the
    embedding model can attend to section boundaries.
    """
    lines: list[str] = []

    # ── Section 1: Identity ──────────────────────────────────────────────────
    lines.append(f"Dish: {record.name}")
    lines.append(
        f"Cuisine: {record.cuisine} | Meal type: {record.meal_type}"
    )

    # ── Section 2: Mood tags ─────────────────────────────────────────────────
    if record.moods:
        lines.append(f"Moods: {', '.join(record.moods)}")

    # ── Section 3: Ingredients ───────────────────────────────────────────────
    if record.ingredients:
        lines.append(f"Ingredients: {', '.join(record.ingredients)}")

    # ── Section 4: Dietary & spice ───────────────────────────────────────────
    dietary_part = (
        f"Dietary: {', '.join(record.dietary_tags)}"
        if record.dietary_tags
        else "Dietary: none specified"
    )
    lines.append(f"{dietary_part} | Spice: {record.spice_level}")

    # ── Section 5: Cooking characteristics ──────────────────────────────────
    lines.append(
        f"Cooking method: {record.cooking_method} | "
        f"Flavour: {record.flavour_profile} | "
        f"Texture: {record.texture}"
    )

    # ── Section 6: Context & nutrition ───────────────────────────────────────
    lines.append(
        f"Occasion: {record.occasion} | "
        f"Prep time: {record.prep_time_mins} min | "
        f"Approx. {record.calories_approx} kcal | "
        f"Serves {record.servings}"
    )

    # ── Section 7: Free-text description ─────────────────────────────────────
    if record.description:
        lines.append(f"Description: {record.description}")

    # ── Section 8: Mood descriptor expansion ────────────────────────────────
    # For each mood on this record that exists in the mood_mapping,
    # append its descriptor keywords. This means a chunk for a "stressed"
    # dish also contains words like "calming", "comfort food", "indulgent" —
    # strongly boosting retrieval when the user says "I'm feeling stressed".
    expanded_descriptors: list[str] = []
    for mood in record.moods:
        descriptors = mood_mapping.get(mood.strip().lower(), [])
        expanded_descriptors.extend(descriptors)

    if expanded_descriptors:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_desc: list[str] = []
        for d in expanded_descriptors:
            if d not in seen:
                seen.add(d)
                unique_desc.append(d)
        lines.append(f"Mood descriptors: {', '.join(unique_desc)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_metadata(record: FoodRecord) -> dict:
    """
    Build the metadata dict stored alongside each vector in the DB.

    Rules
    ─────
    - All values must be scalar (str, int, float, bool) — no lists.
      ChromaDB and FAISS metadata stores don't support nested types.
    - Lists are joined to comma-separated strings.
    - The metadata must contain everything needed to render a food card
      in the UI so the app never needs to re-read the CSV at query time.
    """
    return {
        # Identity
        "chunk_id":       record.id,
        "name":           record.name,
        "cuisine":        record.cuisine,
        "meal_type":      record.meal_type,

        # Display content
        "description":    record.description,
        "ingredients":    ", ".join(record.ingredients),

        # Mood & diet (used for filtering)
        "moods":          ", ".join(record.moods),
        "dietary_tags":   ", ".join(record.dietary_tags),
        "spice_level":    record.spice_level,

        # Numeric attributes
        "prep_time_mins":  record.prep_time_mins,
        "calories_approx": record.calories_approx,
        "servings":        record.servings,

        # Characteristics
        "cooking_method":  record.cooking_method,
        "flavour_profile": record.flavour_profile,
        "texture":         record.texture,
        "occasion":        record.occasion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """
    Serialise chunks to a JSON file at ``path``.

    The file is a JSON array of chunk dicts:
        [{"chunk_id": "...", "document": "...", "metadata": {...}}, ...]

    Args:
        chunks: List of Chunk objects to save.
        path:   Destination file path (parent directories are created).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [c.to_dict() for c in chunks]

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(
        f"Saved {len(chunks):,} chunks → '{path}' ({size_mb:.1f} MB)"
    )


def load_chunks(path: str | Path) -> list[Chunk]:
    """
    Deserialise chunks from a previously saved JSON file.

    Args:
        path: Path to a chunks.json file written by ``save_chunks()``.

    Returns:
        List of Chunk objects.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at '{path}'. "
            "Run the ingestion pipeline first: make ingest"
        )

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    chunks = [Chunk.from_dict(d) for d in data]
    logger.info(f"Loaded {len(chunks):,} chunks from '{path}'.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Stats helper (used by ingest.py for progress reporting)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_stats(chunks: list[Chunk]) -> dict:
    """
    Compute summary statistics over a list of chunks.
    Returned dict is logged by ingest.py after chunking completes.
    """
    if not chunks:
        return {"total": 0}

    doc_lengths = [len(c.document) for c in chunks]
    token_estimates = [len(c.document.split()) for c in chunks]

    # Cuisine distribution
    from collections import Counter
    cuisine_counts = Counter(
        c.metadata.get("cuisine", "unknown") for c in chunks
    )
    top_cuisines = dict(cuisine_counts.most_common(5))

    # Mood coverage
    all_moods: list[str] = []
    for c in chunks:
        moods_str = c.metadata.get("moods", "")
        if moods_str:
            all_moods.extend(m.strip() for m in moods_str.split(","))
    mood_counts = Counter(all_moods)

    return {
        "total_chunks":          len(chunks),
        "avg_doc_chars":         round(sum(doc_lengths) / len(doc_lengths)),
        "min_doc_chars":         min(doc_lengths),
        "max_doc_chars":         max(doc_lengths),
        "avg_token_estimate":    round(sum(token_estimates) / len(token_estimates)),
        "top_5_cuisines":        top_cuisines,
        "unique_moods_covered":  len(mood_counts),
        "top_5_moods":           dict(mood_counts.most_common(5)),
    }