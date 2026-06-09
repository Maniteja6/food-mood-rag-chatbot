"""
ingestion/ingest.py
═══════════════════
Main orchestrator for the MoodBite data ingestion pipeline.

Run once before starting the app to build the vector database:

    python -m ingestion.ingest            # uses .env for all config
    python -m ingestion.ingest --force    # wipe and rebuild existing DB
    python -m ingestion.ingest --dry-run  # validate + chunk, skip embedding
    python -m ingestion.ingest --help     # show all options

Or via Make:
    make ingest
    make ingest-force

Pipeline stages
───────────────
  Stage 1 — Load
      Read food_dataset.csv and mood_food_mapping.json from disk.
      Validate every row; skip bad rows with a warning.

  Stage 2 — Chunk
      Convert each FoodRecord into an embeddable Chunk (one per record).
      Build rich document strings that pack mood tags, ingredients,
      dietary info, and free-text description into one dense block.
      Save processed chunks to data/processed/chunks.json.

  Stage 3 — Embed + Store
      Batch-embed all chunk documents using the configured embedding model
      (OpenAI text-embedding-3-small or HuggingFace sentence-transformers).
      Write vectors + metadata to the vector database (ChromaDB or FAISS).
      Skips records already present in the DB unless --force is passed.

Environment variables (read from .env)
───────────────────────────────────────
  FOOD_DATASET_PATH         path to food_dataset.csv
  MOOD_MAPPING_PATH         path to mood_food_mapping.json
  PROCESSED_CHUNKS_PATH     path to save chunks.json
  EMBEDDING_PROVIDER        openai | huggingface
  EMBEDDING_MODEL           e.g. text-embedding-3-small
  HF_EMBEDDING_MODEL        e.g. sentence-transformers/all-MiniLM-L6-v2
  VECTOR_STORE_PROVIDER     chroma | faiss
  VECTOR_DB_PATH            directory to persist the vector DB
  CHROMA_COLLECTION_NAME    collection name (ChromaDB only)
  OPENAI_API_KEY            required when EMBEDDING_PROVIDER=openai
  LOG_LEVEL                 DEBUG | INFO | WARNING
  LOG_FILE                  path to write log file
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Ensure repo root is on sys.path so `python -m ingestion.ingest` works ──
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Load .env before importing anything that reads os.environ ──────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)

from ingestion.loader  import load_food_dataset, load_mood_mapping, FoodRecord
from ingestion.chunker import chunk_records, save_chunks, load_chunks, chunk_stats, Chunk


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level_str: str = "INFO", log_file: str | None = None) -> logging.Logger:
    level = getattr(logging, level_str.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("ingestion.ingest")


# ─────────────────────────────────────────────────────────────────────────────
# Config reader
# ─────────────────────────────────────────────────────────────────────────────

def _get_config() -> dict[str, Any]:
    """Read all ingestion-relevant env vars with sensible defaults."""
    return {
        "food_dataset_path":     os.getenv("FOOD_DATASET_PATH",        "./data/raw/food_dataset.csv"),
        "mood_mapping_path":     os.getenv("MOOD_MAPPING_PATH",        "./data/raw/mood_food_mapping.json"),
        "processed_chunks_path": os.getenv("PROCESSED_CHUNKS_PATH",    "./data/processed/chunks.json"),
        "embedding_provider":    os.getenv("EMBEDDING_PROVIDER",       "openai").lower(),
        "embedding_model":       os.getenv("EMBEDDING_MODEL",          "text-embedding-3-small"),
        "hf_embedding_model":    os.getenv("HF_EMBEDDING_MODEL",       "sentence-transformers/all-MiniLM-L6-v2"),
        "vector_store_provider": os.getenv("VECTOR_STORE_PROVIDER",    "chroma").lower(),
        "vector_db_path":        os.getenv("VECTOR_DB_PATH",           "./data/vector_db"),
        "chroma_collection":     os.getenv("CHROMA_COLLECTION_NAME",   "food_mood_collection"),
        "openai_api_key":        os.getenv("OPENAI_API_KEY",           ""),
        "log_level":             os.getenv("LOG_LEVEL",                "INFO"),
        "log_file":              os.getenv("LOG_FILE",                 "./logs/app.log"),
        "batch_size":            int(os.getenv("EMBED_BATCH_SIZE",     "100")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Load
# ─────────────────────────────────────────────────────────────────────────────

def stage_load(cfg: dict, log: logging.Logger) -> tuple[list[FoodRecord], dict[str, list[str]]]:
    """Load and validate the food dataset and mood mapping."""
    _banner(log, "STAGE 1 — LOAD")
    t0 = time.perf_counter()

    records = load_food_dataset(cfg["food_dataset_path"])
    log.info(f"  ✓ Food dataset : {len(records):,} records loaded")

    mood_mapping: dict[str, list[str]] = {}
    mood_path = Path(cfg["mood_mapping_path"])
    if mood_path.exists():
        mood_mapping = load_mood_mapping(cfg["mood_mapping_path"])
        log.info(f"  ✓ Mood mapping : {len(mood_mapping)} moods loaded")
    else:
        log.warning(
            f"  ⚠  Mood mapping not found at '{mood_path}'. "
            "Chunking will proceed without mood descriptor expansion."
        )

    log.info(f"  Stage 1 done in {time.perf_counter() - t0:.1f}s")
    return records, mood_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Chunk
# ─────────────────────────────────────────────────────────────────────────────

def stage_chunk(
    records: list[FoodRecord],
    mood_mapping: dict[str, list[str]],
    cfg: dict,
    log: logging.Logger,
    force: bool = False,
) -> list[Chunk]:
    """
    Convert records to chunks and save to disk.
    If chunks.json already exists and --force is not set, load from disk
    instead of re-chunking (saves time on repeated runs).
    """
    _banner(log, "STAGE 2 — CHUNK")
    t0 = time.perf_counter()

    chunks_path = Path(cfg["processed_chunks_path"])

    if chunks_path.exists() and not force:
        log.info(f"  Found existing chunks at '{chunks_path}' — loading from cache.")
        log.info("  (Pass --force to re-chunk from scratch.)")
        chunks = load_chunks(chunks_path)
        log.info(f"  ✓ Loaded {len(chunks):,} cached chunks.")
    else:
        log.info(f"  Chunking {len(records):,} records …")
        chunks = chunk_records(records, mood_mapping)

        stats = chunk_stats(chunks)
        log.info(f"  ✓ {stats['total_chunks']:,} chunks created")
        log.info(f"     Avg document length : {stats['avg_doc_chars']} chars "
                 f"(~{stats['avg_token_estimate']} tokens)")
        log.info(f"     Top cuisines        : {stats['top_5_cuisines']}")
        log.info(f"     Unique moods        : {stats['unique_moods_covered']}")
        log.info(f"     Top moods           : {stats['top_5_moods']}")

        log.info(f"  Saving chunks → '{chunks_path}' …")
        save_chunks(chunks, chunks_path)

    log.info(f"  Stage 2 done in {time.perf_counter() - t0:.1f}s")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Embed + Store
# ─────────────────────────────────────────────────────────────────────────────

def stage_embed_and_store(
    chunks: list[Chunk],
    cfg: dict,
    log: logging.Logger,
    force: bool = False,
) -> None:
    """
    Embed all chunks and upsert into the vector store.

    Embedding providers
    ───────────────────
    openai      : Uses OpenAI's Embeddings API in batches. Requires
                  OPENAI_API_KEY in .env. Each batch is retried up to 3
                  times with exponential back-off on rate-limit errors.

    huggingface : Runs sentence-transformers locally (no API key needed).
                  Slower but free and works offline / on Streamlit Cloud
                  without secrets.

    Vector store backends
    ──────────────────────
    chroma      : Persistent ChromaDB collection in VECTOR_DB_PATH.
                  Supports metadata filtering. Good default choice.

    faiss       : Flat FAISS index saved as two files (index.faiss +
                  metadata.json). No filtering support but extremely fast
                  for pure similarity search.
    """
    _banner(log, "STAGE 3 — EMBED + STORE")
    t0 = time.perf_counter()

    provider = cfg["embedding_provider"]
    store    = cfg["vector_store_provider"]

    log.info(f"  Embedding provider : {provider}")
    log.info(f"  Vector store       : {store}")
    log.info(f"  Total chunks       : {len(chunks):,}")

    # ── Build embedding function ─────────────────────────────────────────────
    embed_fn = _build_embedding_fn(provider, cfg, log)

    # ── Build / open vector store ────────────────────────────────────────────
    vector_store = _build_vector_store(store, cfg, log, force)

    # ── Check which chunks already exist (skip on non-force runs) ───────────
    chunks_to_insert = _filter_existing(chunks, vector_store, store, force, log)

    if not chunks_to_insert:
        log.info("  ✓ All chunks already present in vector store — nothing to do.")
        log.info("    (Pass --force to re-embed and overwrite.)")
    else:
        log.info(f"  Embedding and inserting {len(chunks_to_insert):,} chunks …")
        _embed_and_upsert(chunks_to_insert, embed_fn, vector_store, store, cfg, log)

    log.info(f"  Stage 3 done in {time.perf_counter() - t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding function builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_embedding_fn(
    provider: str,
    cfg: dict,
    log: logging.Logger,
):
    """Return a callable: list[str] → list[list[float]]"""

    if provider == "openai":
        return _build_openai_embed_fn(cfg, log)
    elif provider == "huggingface":
        return _build_hf_embed_fn(cfg, log)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. "
            "Choose 'openai' or 'huggingface'."
        )


def _build_openai_embed_fn(cfg: dict, log: logging.Logger):
    """Build an OpenAI embedding function with retry logic."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not found. Run: pip install openai"
        )

    api_key = cfg["openai_api_key"]
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    client = OpenAI(api_key=api_key)
    model  = cfg["embedding_model"]
    log.info(f"  OpenAI embedding model : {model}")

    def embed(texts: list[str]) -> list[list[float]]:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = client.embeddings.create(
                    input=texts,
                    model=model,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:                            # noqa: BLE001
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt
                log.warning(
                    f"  OpenAI embed error (attempt {attempt}/{max_retries}): "
                    f"{exc}. Retrying in {wait}s …"
                )
                time.sleep(wait)

    return embed


def _build_hf_embed_fn(cfg: dict, log: logging.Logger):
    """Build a local HuggingFace sentence-transformers embedding function."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not found. Run: pip install sentence-transformers"
        )

    model_name = cfg["hf_embedding_model"]
    log.info(f"  Loading HuggingFace model '{model_name}' …")
    model = SentenceTransformer(model_name)
    log.info("  ✓ HuggingFace model loaded.")

    def embed(texts: list[str]) -> list[list[float]]:
        embeddings = model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    return embed


# ─────────────────────────────────────────────────────────────────────────────
# Vector store builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_vector_store(
    provider: str,
    cfg: dict,
    log: logging.Logger,
    force: bool,
) -> Any:
    """Open or create the vector store. Returns the store object."""

    db_path = Path(cfg["vector_db_path"])
    db_path.mkdir(parents=True, exist_ok=True)

    if provider == "chroma":
        return _build_chroma_store(cfg, log, force)
    elif provider == "faiss":
        return _build_faiss_store(cfg, log, force)
    else:
        raise ValueError(
            f"Unknown VECTOR_STORE_PROVIDER '{provider}'. "
            "Choose 'chroma' or 'faiss'."
        )


def _build_chroma_store(cfg: dict, log: logging.Logger, force: bool) -> Any:
    try:
        import chromadb
    except ImportError:
        raise ImportError("chromadb not found. Run: pip install chromadb")

    db_path    = cfg["vector_db_path"]
    collection = cfg["chroma_collection"]

    client = chromadb.PersistentClient(path=str(db_path))

    if force:
        try:
            client.delete_collection(collection)
            log.info(f"  Deleted existing ChromaDB collection '{collection}'.")
        except Exception:                                       # noqa: BLE001
            pass

    col = client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},
    )
    existing_count = col.count()
    log.info(
        f"  ✓ ChromaDB collection '{collection}' opened "
        f"({existing_count:,} existing vectors)."
    )
    return col


def _build_faiss_store(cfg: dict, log: logging.Logger, force: bool) -> dict:
    """
    FAISS store is represented as a dict with keys:
        'index'    : faiss.IndexFlatIP  (inner product → cosine after L2 norm)
        'meta'     : list[dict]         metadata parallel array
        'path'     : Path               directory to save/load from
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        raise ImportError("faiss-cpu not found. Run: pip install faiss-cpu")

    db_path    = Path(cfg["vector_db_path"])
    index_file = db_path / "index.faiss"
    meta_file  = db_path / "metadata.json"

    if force and index_file.exists():
        index_file.unlink()
        meta_file.unlink(missing_ok=True)
        log.info("  Deleted existing FAISS index files.")

    if index_file.exists() and meta_file.exists():
        log.info(f"  Loading existing FAISS index from '{db_path}' …")
        index = faiss.read_index(str(index_file))
        with open(meta_file, encoding="utf-8") as fh:
            meta = json.load(fh)
        log.info(f"  ✓ FAISS index loaded ({index.ntotal:,} vectors).")
    else:
        # Defer dimension until first batch is embedded
        index = None
        meta  = []
        log.info("  FAISS index will be created on first embedding batch.")

    return {"index": index, "meta": meta, "path": db_path}


# ─────────────────────────────────────────────────────────────────────────────
# Existing-record filter
# ─────────────────────────────────────────────────────────────────────────────

def _filter_existing(
    chunks: list[Chunk],
    vector_store: Any,
    store_provider: str,
    force: bool,
    log: logging.Logger,
) -> list[Chunk]:
    """Return only the chunks that are NOT yet in the vector store."""
    if force:
        return chunks

    if store_provider == "chroma":
        existing_ids = set(vector_store.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        skipped = len(chunks) - len(new_chunks)
        if skipped:
            log.info(f"  Skipping {skipped:,} chunks already in ChromaDB.")
        return new_chunks

    elif store_provider == "faiss":
        existing_ids = {m["chunk_id"] for m in vector_store["meta"]}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        skipped = len(chunks) - len(new_chunks)
        if skipped:
            log.info(f"  Skipping {skipped:,} chunks already in FAISS index.")
        return new_chunks

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Core embed + upsert loop
# ─────────────────────────────────────────────────────────────────────────────

def _embed_and_upsert(
    chunks: list[Chunk],
    embed_fn,
    vector_store: Any,
    store_provider: str,
    cfg: dict,
    log: logging.Logger,
) -> None:
    """
    Embed chunks in batches and write to the vector store.
    Logs progress every batch with elapsed time and ETA.
    """
    import json as _json  # local import to avoid shadowing module-level name

    batch_size = cfg["batch_size"]
    total      = len(chunks)
    inserted   = 0
    t_start    = time.perf_counter()

    for batch_start in range(0, total, batch_size):
        batch      = chunks[batch_start : batch_start + batch_size]
        documents  = [c.document  for c in batch]
        chunk_ids  = [c.chunk_id  for c in batch]
        metadatas  = [c.metadata  for c in batch]

        # ── Embed ────────────────────────────────────────────────────────────
        vectors = embed_fn(documents)

        # ── Upsert into store ────────────────────────────────────────────────
        if store_provider == "chroma":
            vector_store.upsert(
                ids=chunk_ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
            )

        elif store_provider == "faiss":
            import faiss
            import numpy as np

            mat = np.array(vectors, dtype="float32")
            # L2-normalise so inner-product == cosine similarity
            faiss.normalize_L2(mat)

            if vector_store["index"] is None:
                dim = mat.shape[1]
                vector_store["index"] = faiss.IndexFlatIP(dim)
                log.info(f"  Created FAISS IndexFlatIP with dim={dim}.")

            vector_store["index"].add(mat)
            vector_store["meta"].extend(metadatas)

        inserted += len(batch)

        # ── Progress log ─────────────────────────────────────────────────────
        elapsed   = time.perf_counter() - t_start
        pct       = inserted / total * 100
        rate      = inserted / elapsed if elapsed > 0 else 0
        remaining = (total - inserted) / rate if rate > 0 else 0
        log.info(
            f"  [{inserted:>6,}/{total:,}]  {pct:5.1f}%  "
            f"{rate:6.0f} chunks/s  ETA {remaining:5.0f}s"
        )

    # ── Persist FAISS to disk ────────────────────────────────────────────────
    if store_provider == "faiss" and vector_store["index"] is not None:
        import faiss
        db_path    = vector_store["path"]
        index_file = db_path / "index.faiss"
        meta_file  = db_path / "metadata.json"

        faiss.write_index(vector_store["index"], str(index_file))
        with open(meta_file, "w", encoding="utf-8") as fh:
            _json.dump(vector_store["meta"], fh, ensure_ascii=False)

        log.info(
            f"  ✓ FAISS index saved → '{index_file}' "
            f"({vector_store['index'].ntotal:,} vectors total)"
        )

    total_time = time.perf_counter() - t_start
    log.info(
        f"  ✓ Inserted {inserted:,} vectors in {total_time:.1f}s "
        f"({inserted / total_time:.0f} chunks/s avg)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run validator
# ─────────────────────────────────────────────────────────────────────────────

def stage_dry_run(
    records: list[FoodRecord],
    chunks: list[Chunk],
    cfg: dict,
    log: logging.Logger,
) -> None:
    """
    Print a validation report without touching the vector store.
    Checks embeddings can be imported and prints sample chunk text.
    """
    _banner(log, "DRY RUN — VALIDATION REPORT")

    log.info(f"  Records loaded      : {len(records):,}")
    log.info(f"  Chunks created      : {len(chunks):,}")

    stats = chunk_stats(chunks)
    log.info(f"  Avg doc length      : {stats['avg_doc_chars']} chars "
             f"(~{stats['avg_token_estimate']} tokens)")
    log.info(f"  Min / Max doc chars : {stats['min_doc_chars']} / {stats['max_doc_chars']}")
    log.info(f"  Cuisines (top 5)    : {stats['top_5_cuisines']}")
    log.info(f"  Mood coverage       : {stats['unique_moods_covered']} moods")

    # ── Sample chunk ─────────────────────────────────────────────────────────
    log.info("")
    log.info("  ── Sample chunk (first record) ──────────────────────────")
    sample = chunks[0]
    log.info(f"  chunk_id : {sample.chunk_id}")
    for line in sample.document.split("\n"):
        log.info(f"  | {line}")
    log.info("")

    # ── Embedding import check ───────────────────────────────────────────────
    provider = cfg["embedding_provider"]
    log.info(f"  ── Embedding provider check: {provider} ─────────────────")
    if provider == "openai":
        try:
            import openai
            log.info(f"  ✓ openai {openai.__version__} importable")
            if not cfg["openai_api_key"]:
                log.warning("  ⚠  OPENAI_API_KEY is not set — embedding will fail at runtime.")
            else:
                log.info("  ✓ OPENAI_API_KEY is set")
        except ImportError:
            log.error("  ✗ openai package not installed — run: pip install openai")
    elif provider == "huggingface":
        try:
            import sentence_transformers
            log.info(f"  ✓ sentence-transformers {sentence_transformers.__version__} importable")
        except ImportError:
            log.error("  ✗ sentence-transformers not installed — run: pip install sentence-transformers")

    # ── Vector store import check ────────────────────────────────────────────
    store = cfg["vector_store_provider"]
    log.info(f"  ── Vector store check: {store} ──────────────────────────")
    if store == "chroma":
        try:
            import chromadb
            log.info(f"  ✓ chromadb {chromadb.__version__} importable")
        except ImportError:
            log.error("  ✗ chromadb not installed — run: pip install chromadb")
    elif store == "faiss":
        try:
            import faiss
            log.info("  ✓ faiss importable")
        except ImportError:
            log.error("  ✗ faiss-cpu not installed — run: pip install faiss-cpu")

    log.info("")
    log.info("  Dry run complete — no vectors written.")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _banner(log: logging.Logger, title: str) -> None:
    bar = "─" * 60
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.ingest",
        description="MoodBite — Food RAG ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ingestion.ingest                # normal run
  python -m ingestion.ingest --force        # wipe DB and rebuild
  python -m ingestion.ingest --dry-run      # validate, no embedding
  python -m ingestion.ingest --batch-size 50
        """,
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Delete existing vector DB and processed chunks, then rebuild from scratch.",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        dest="dry_run",
        help="Load and chunk data, print a validation report, but skip embedding/storing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override EMBED_BATCH_SIZE env var (default: 100).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override LOG_LEVEL env var.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    cfg  = _get_config()

    # CLI overrides
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.log_level:
        cfg["log_level"] = args.log_level

    log = _setup_logging(cfg["log_level"], cfg["log_file"])

    # ── Pipeline header ──────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  MoodBite — Ingestion Pipeline")
    log.info("═" * 60)
    log.info(f"  Embedding provider : {cfg['embedding_provider']}")
    log.info(f"  Vector store       : {cfg['vector_store_provider']}")
    log.info(f"  Dataset path       : {cfg['food_dataset_path']}")
    log.info(f"  Vector DB path     : {cfg['vector_db_path']}")
    log.info(f"  Batch size         : {cfg['batch_size']}")
    log.info(f"  Force rebuild      : {args.force}")
    log.info(f"  Dry run            : {args.dry_run}")
    log.info("")

    pipeline_start = time.perf_counter()

    try:
        # ── Stage 1: Load ────────────────────────────────────────────────────
        records, mood_mapping = stage_load(cfg, log)

        # ── Stage 2: Chunk ───────────────────────────────────────────────────
        chunks = stage_chunk(records, mood_mapping, cfg, log, force=args.force)

        # ── Stage 3: Embed + Store (skip in dry-run mode) ────────────────────
        if args.dry_run:
            stage_dry_run(records, chunks, cfg, log)
        else:
            stage_embed_and_store(chunks, cfg, log, force=args.force)

    except FileNotFoundError as exc:
        log.error(f"File not found: {exc}")
        log.error("Make sure you have run the dataset generator: python generate_dataset.py")
        sys.exit(1)
    except ValueError as exc:
        log.error(f"Configuration or data error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Ingestion interrupted by user.")
        sys.exit(130)
    except Exception as exc:                                    # noqa: BLE001
        log.exception(f"Unexpected error: {exc}")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start
    log.info("")
    log.info("═" * 60)
    log.info(f"  ✓ Ingestion complete in {total_time:.1f}s")
    log.info(f"  Vector DB → {cfg['vector_db_path']}")
    log.info(f"  Chunks    → {cfg['processed_chunks_path']}")
    log.info("  Run 'streamlit run app/main.py' to start the app.")
    log.info("═" * 60)


if __name__ == "__main__":
    # Also importable as: from ingestion.ingest import main
    import json  # ensure available for FAISS path
    main()