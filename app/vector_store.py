"""
vector_store.py — FAISS layer for storing and searching face embeddings (512D).

The system relies on:
- IndexFlatIP: Inner Product — equivalent to Cosine Similarity after L2 normalization.
- IndexIDMap: To link each embedding to its own ID (same ID as the SQLite table).
- Immediate disk persistence after each addition to prevent data loss.
"""
import os
import threading

import faiss
import numpy as np

INDEX_PATH = os.path.join("models", "faces.index")
EMBEDDING_DIM = 512

# Lock for concurrent writes
_lock = threading.Lock()

# In-memory index (None until initialized)
_index: faiss.IndexIDMap | None = None


def _build_empty_index() -> faiss.IndexIDMap:
    """Build a new empty FAISS index."""
    flat = faiss.IndexFlatIP(EMBEDDING_DIM)
    return faiss.IndexIDMap(flat)


def init_index():
    """
    Load the index from disk if it exists, or create a new one.
    Called once at server startup.
    """
    global _index
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
    else:
        _index = _build_empty_index()


def _save_index():
    """Save the current index to disk (called within a lock)."""
    faiss.write_index(_index, INDEX_PATH)


def add_embedding(person_id: int, embedding: np.ndarray):
    """
    Add a new face embedding to the index and save immediately.

    Args:
        person_id: The person's ID in SQLite.
        embedding: numpy array of shape (512,), will be L2-normalized by FAISS.
    """
    global _index
    vec = embedding.astype(np.float32).reshape(1, EMBEDDING_DIM)
    faiss.normalize_L2(vec)  # Exact L2 normalization — same image always scores 1.0
    ids = np.array([person_id], dtype=np.int64)
    with _lock:
        _index.add_with_ids(vec, ids)
        _save_index()


def search(embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Search for the closest faces to an unknown embedding.

    Args:
        embedding: numpy array of shape (512,), L2-normalized.
        top_k: Number of results to return.

    Returns:
        List of dicts: [{"person_id": int, "score": float}, ...]
        Sorted in descending order by similarity (1.0 = exact match).
    """
    global _index
    if _index is None or _index.ntotal == 0:
        return []

    k = min(top_k, _index.ntotal)
    vec = embedding.astype(np.float32).reshape(1, EMBEDDING_DIM)
    faiss.normalize_L2(vec)  # Exact L2 normalization — same image always scores 1.0

    scores, ids = _index.search(vec, k)

    results = []
    for score, pid in zip(scores[0], ids[0]):
        if pid == -1:          # FAISS returns -1 for unavailable results
            continue
        results.append({
            "person_id": int(pid),
            "score": float(round(score, 4)),
        })
    return results


def remove_embedding(person_id: int):
    """
    Remove an embedding from the index using remove_ids.
    """
    global _index
    with _lock:
        if _index is None or _index.ntotal == 0:
            return

        n_before = _index.ntotal
        _index.remove_ids(np.array([person_id], dtype=np.int64))
        if _index.ntotal < n_before:
            _save_index()


def get_total() -> int:
    """Return the number of currently indexed faces."""
    if _index is None:
        return 0
    return int(_index.ntotal)


def reload_index():
    """Reload the FAISS index from disk. Called after a successful restore."""
    global _index
    with _lock:
        if os.path.exists(INDEX_PATH):
            _index = faiss.read_index(INDEX_PATH)
        else:
            _index = _build_empty_index()


def load_all_embeddings(index_path: str) -> dict[int, np.ndarray]:
    """
    Read all (person_id → embedding) pairs from a FAISS index file on disk.
    Used during merge-restore to extract vectors from the backup index.
    """
    idx = faiss.read_index(index_path)
    if idx.ntotal == 0:
        return {}
    ids = faiss.vector_to_array(idx.id_map)                        # shape (n,) int64
    flat = faiss.downcast_index(idx.index)                         # IndexFlatIP
    vecs = faiss.vector_to_array(flat.xb).reshape(idx.ntotal, EMBEDDING_DIM)
    return {int(ids[i]): vecs[i].copy() for i in range(idx.ntotal)}


def add_embeddings_batch(items: list[tuple[int, np.ndarray]]):
    """
    Add multiple (person_id, embedding) pairs in a single lock+save.
    More efficient than calling add_embedding() in a loop.
    """
    global _index
    if not items:
        return
    vecs = np.array([item[1].astype(np.float32) for item in items], dtype=np.float32)
    faiss.normalize_L2(vecs)
    ids = np.array([item[0] for item in items], dtype=np.int64)
    with _lock:
        _index.add_with_ids(vecs, ids)
        _save_index()
