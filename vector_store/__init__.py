"""
vector_store — Vector Database Abstraction Layer
════════════════════════════════════════════════

Provides a provider-agnostic interface for reading and writing dense
float vectors with associated metadata.

Supported backends
──────────────────
chroma  (default)
    ChromaDB PersistentClient.  Automatic disk persistence, cosine similarity
    natively, metadata filtering support.  Best for most deployments.

faiss
    FAISS IndexFlatIP with a JSON metadata sidecar.  Lightest dependency,
    fastest raw ANN search.  No metadata filtering support.
    Use when chromadb causes dependency conflicts.

Quick start
───────────
    # Read .env and return the correct backend automatically
    from vector_store import get_store

    store = get_store()                           # from .env
    store = get_store(provider="chroma")          # explicit
    store = get_store(provider="faiss")

    # Or use the classes directly
    from vector_store import ChromaStore, FAISSStore, VectorStoreBase

    store = ChromaStore.from_settings()
    store.upsert(ids, vectors, documents, metadatas)
    results = store.query(query_vector, top_k=5)

Public symbols
──────────────
    VectorStoreBase    Abstract base class defining the interface
    ChromaStore        ChromaDB implementation
    FAISSStore         FAISS implementation
    QueryResult        TypedDict returned by store.query()
    get_store()        Factory convenience function
"""

from vector_store.base        import VectorStoreBase, QueryResult
from vector_store.chroma_store import ChromaStore
from vector_store.faiss_store  import FAISSStore


def get_store(
    provider: str | None = None,
    settings=None,
) -> VectorStoreBase:
    """
    Return a configured vector store for the given provider.

    Args:
        provider: "chroma" | "faiss" | None.
                  If None, reads VECTOR_STORE_PROVIDER from .env via Settings.
        settings: Optional Settings instance.  Reads .env if None.

    Returns:
        Configured ChromaStore or FAISSStore instance.

    Examples:
        store = get_store()                  # from .env
        store = get_store("chroma")          # explicit ChromaDB
        store = get_store("faiss")           # explicit FAISS
    """
    if provider is not None:
        # Explicit override — build with default settings
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        if provider.lower() == "chroma":
            return ChromaStore.from_settings(settings)
        elif provider.lower() == "faiss":
            return FAISSStore.from_settings(settings)
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose 'chroma' or 'faiss'."
            )

    return VectorStoreBase.from_settings(settings)


__all__ = [
    "VectorStoreBase",
    "ChromaStore",
    "FAISSStore",
    "QueryResult",
    "get_store",
]