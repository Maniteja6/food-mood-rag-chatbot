"""
ingestion/loader.py
═══════════════════
Responsible for loading raw source data from disk and returning clean,
validated Python objects ready for chunking.

Public API
──────────
    load_food_dataset(path)      → list[FoodRecord]
    load_mood_mapping(path)      → dict[str, list[str]]
    FoodRecord                   dataclass

Design notes
────────────
- Every row is validated field-by-field; bad rows are logged and skipped
  rather than crashing the whole pipeline.
- All string fields are stripped and normalised so downstream code never
  has to worry about leading/trailing whitespace or inconsistent casing.
- The loader is stateless and has no side-effects beyond reading files.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoodRecord:
    """
    A single validated food entry from the CSV dataset.
    All list fields are already split and stripped — no further parsing needed.
    """
    id:              str
    name:            str
    cuisine:         str
    meal_type:       str
    description:     str
    ingredients:     list[str]
    moods:           list[str]
    dietary_tags:    list[str]
    spice_level:     str
    prep_time_mins:  int
    calories_approx: int
    servings:        int
    cooking_method:  str
    flavour_profile: str
    texture:         str
    occasion:        str

    # ── Convenience helpers ──────────────────────────────────────────────────

    def to_metadata(self) -> dict:
        """
        Return a flat dict of scalar fields suitable for vector-store metadata.
        Lists are serialised to comma-separated strings because most vector
        stores (ChromaDB, FAISS+JSON) only accept scalar metadata values.
        """
        return {
            "id":              self.id,
            "name":            self.name,
            "cuisine":         self.cuisine,
            "meal_type":       self.meal_type,
            "ingredients":     ", ".join(self.ingredients),
            "moods":           ", ".join(self.moods),
            "dietary_tags":    ", ".join(self.dietary_tags),
            "spice_level":     self.spice_level,
            "prep_time_mins":  self.prep_time_mins,
            "calories_approx": self.calories_approx,
            "servings":        self.servings,
            "cooking_method":  self.cooking_method,
            "flavour_profile": self.flavour_profile,
            "texture":         self.texture,
            "occasion":        self.occasion,
        }

    def to_dict(self) -> dict:
        """Full dict including list fields (for JSON serialisation)."""
        return {
            "id":              self.id,
            "name":            self.name,
            "cuisine":         self.cuisine,
            "meal_type":       self.meal_type,
            "description":     self.description,
            "ingredients":     self.ingredients,
            "moods":           self.moods,
            "dietary_tags":    self.dietary_tags,
            "spice_level":     self.spice_level,
            "prep_time_mins":  self.prep_time_mins,
            "calories_approx": self.calories_approx,
            "servings":        self.servings,
            "cooking_method":  self.cooking_method,
            "flavour_profile": self.flavour_profile,
            "texture":         self.texture,
            "occasion":        self.occasion,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Required columns
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_COLUMNS = {
    "id", "name", "cuisine", "meal_type", "description",
    "ingredients", "moods", "prep_time_mins",
}

_OPTIONAL_DEFAULTS: dict[str, str] = {
    "dietary_tags":    "",
    "spice_level":     "Unknown",
    "calories_approx": "0",
    "servings":        "1",
    "cooking_method":  "unknown",
    "flavour_profile": "unknown",
    "texture":         "unknown",
    "occasion":        "any",
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_food_dataset(path: str | Path) -> list[FoodRecord]:
    """
    Load and validate the food dataset CSV.

    Args:
        path: Path to food_dataset.csv

    Returns:
        List of validated FoodRecord objects (bad rows are skipped).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing from the header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Food dataset not found at '{path}'. "
            "Place food_dataset.csv in data/raw/ before running ingestion."
        )

    logger.info(f"Loading food dataset from '{path}' …")

    records: list[FoodRecord] = []
    skipped = 0

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        # ── Validate header ──────────────────────────────────────────────────
        if reader.fieldnames is None:
            raise ValueError("CSV file appears to be empty.")

        actual_columns = set(reader.fieldnames)
        missing = _REQUIRED_COLUMNS - actual_columns
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}\n"
                f"Found columns: {sorted(actual_columns)}"
            )

        # ── Parse rows ───────────────────────────────────────────────────────
        for line_num, raw in enumerate(reader, start=2):   # line 1 = header
            try:
                record = _parse_row(raw, line_num)
                if record is not None:
                    records.append(record)
            except Exception as exc:                        # noqa: BLE001
                logger.warning(f"Line {line_num}: skipping — {exc}")
                skipped += 1

    logger.info(
        f"Loaded {len(records):,} valid records "
        f"({skipped:,} skipped) from '{path.name}'."
    )
    return records


def _parse_row(raw: dict, line_num: int) -> Optional[FoodRecord]:
    """
    Parse and validate a single CSV row dict into a FoodRecord.
    Returns None if the row should be silently skipped (e.g. blank line).
    Raises ValueError with a clear message for genuinely bad data.
    """
    # Apply optional-column defaults for missing keys
    for col, default in _OPTIONAL_DEFAULTS.items():
        if col not in raw or raw[col] is None or raw[col].strip() == "":
            raw[col] = default

    # ── Required string fields ───────────────────────────────────────────────
    row_id = _require_str(raw, "id", line_num)
    if row_id is None:
        return None  # blank row

    name        = _require_str(raw, "name", line_num) or f"Unnamed_{row_id}"
    cuisine     = _clean_str(raw.get("cuisine", "Unknown"))
    meal_type   = _clean_str(raw.get("meal_type", "Main Course"))
    description = _clean_str(raw.get("description", ""))

    if not description:
        # Build a minimal description from available fields if column is empty
        description = f"{name} — a {cuisine} dish."

    # ── Parsed list fields ───────────────────────────────────────────────────
    ingredients  = _parse_csv_list(raw.get("ingredients", ""))
    moods        = _parse_csv_list(raw.get("moods", ""))
    dietary_tags = _parse_csv_list(raw.get("dietary_tags", ""))

    if not moods:
        moods = ["any"]

    # ── Numeric fields ───────────────────────────────────────────────────────
    prep_time_mins  = _parse_int(raw.get("prep_time_mins",  "0"), default=0)
    calories_approx = _parse_int(raw.get("calories_approx", "0"), default=0)
    servings        = _parse_int(raw.get("servings",        "1"), default=1)

    # ── String enum fields ───────────────────────────────────────────────────
    spice_level     = _clean_str(raw.get("spice_level",     "Unknown"))
    cooking_method  = _clean_str(raw.get("cooking_method",  "unknown"))
    flavour_profile = _clean_str(raw.get("flavour_profile", "unknown"))
    texture         = _clean_str(raw.get("texture",         "unknown"))
    occasion        = _clean_str(raw.get("occasion",        "any"))

    return FoodRecord(
        id=row_id,
        name=name,
        cuisine=cuisine,
        meal_type=meal_type,
        description=description,
        ingredients=ingredients,
        moods=moods,
        dietary_tags=dietary_tags,
        spice_level=spice_level,
        prep_time_mins=prep_time_mins,
        calories_approx=calories_approx,
        servings=servings,
        cooking_method=cooking_method,
        flavour_profile=flavour_profile,
        texture=texture,
        occasion=occasion,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON mood-mapping loader
# ─────────────────────────────────────────────────────────────────────────────

def load_mood_mapping(path: str | Path) -> dict[str, list[str]]:
    """
    Load the mood → food-category keyword mapping from JSON.

    Args:
        path: Path to mood_food_mapping.json

    Returns:
        dict mapping mood name (str) to list of descriptor strings.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the JSON structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Mood mapping not found at '{path}'. "
            "Create data/raw/mood_food_mapping.json before running ingestion."
        )

    logger.info(f"Loading mood mapping from '{path}' …")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"mood_food_mapping.json must be a JSON object (dict), "
            f"got {type(data).__name__}."
        )

    # Normalise: ensure all values are lists of strings
    mapping: dict[str, list[str]] = {}
    for mood, descriptors in data.items():
        mood_key = _clean_str(mood).lower()
        if not mood_key:
            continue
        if isinstance(descriptors, list):
            mapping[mood_key] = [str(d).strip() for d in descriptors if str(d).strip()]
        elif isinstance(descriptors, str):
            mapping[mood_key] = [descriptors.strip()]
        else:
            logger.warning(f"Mood '{mood}': unexpected descriptor type {type(descriptors)}, skipping.")

    logger.info(f"Loaded mood mapping for {len(mapping)} moods.")
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_str(value: object) -> str:
    """Strip and return a string; empty strings stay empty."""
    if value is None:
        return ""
    return str(value).strip()


def _require_str(row: dict, key: str, line_num: int) -> Optional[str]:
    """Return a stripped string or None if blank (silently skip blank rows)."""
    val = _clean_str(row.get(key, ""))
    if not val:
        if key == "id":
            return None   # blank id → blank row, skip silently
        raise ValueError(f"Required column '{key}' is empty.")
    return val


def _parse_csv_list(value: str) -> list[str]:
    """Split a comma-separated string into a list of stripped, non-empty strings."""
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(value: str, default: int = 0) -> int:
    """Parse an integer string; return `default` on failure."""
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default