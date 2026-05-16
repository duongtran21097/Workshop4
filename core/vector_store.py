"""
In-memory vector store using FAISS + SentenceTransformers.
Chunks meeting notes by paragraph, embeds with paraphrase-multilingual-MiniLM-L12-v2,
and indexes with FAISS IndexFlatIP (cosine similarity via L2 normalization).
"""
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

CHUNK_SIZE = 700        # max chars per chunk
CHUNK_OVERLAP = 1       # number of sentences to overlap between chunks

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_CHUNK_LENGTH = 30

_model = None
_chunks: List[str] = []
_metadatas: List[Dict[str, Any]] = []    # {"source": filename, "index": i, "chars": n}
_embeddings: List[List[float]] = []      # kept for visualization (get_all_embeddings)
_index = None                             # faiss.IndexFlatIP, built on store


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def _chunk_text(text: str) -> List[str]:
    """
    Create overlapping chunks from sentences.
    Overlap is sentence-based (CHUNK_OVERLAP sentences) to avoid cutting mid-word.
    """
    sentences = _split_sentences(text)

    chunks = []
    current_sentences: List[str] = []

    for sentence in sentences:
        current_text = " ".join(current_sentences)
        if current_sentences and len(current_text) + len(sentence) > CHUNK_SIZE:
            if len(current_text) >= MIN_CHUNK_LENGTH:
                chunks.append(current_text)
            # overlap: keep last CHUNK_OVERLAP complete sentences
            current_sentences = current_sentences[-CHUNK_OVERLAP:] + [sentence]
        else:
            current_sentences.append(sentence)

    # last chunk
    last = " ".join(current_sentences)
    if len(last.strip()) >= MIN_CHUNK_LENGTH:
        chunks.append(last.strip())

    return chunks



def _encode(texts: List[str]) -> np.ndarray:
    """Encode texts to L2-normalized float32 vectors (cosine via inner product)."""
    embs = _get_model().encode(texts, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embs)
    return embs


def init_store():
    pass


def store_meeting_notes(content: str, source: str = ""):
    """
    Chunk, embed and index meeting notes.
    source: filename or label used to populate chunk metadata.
    """
    global _chunks, _metadatas, _embeddings, _index
    _chunks = _chunk_text(content)
    if not _chunks:
        _metadatas = []
        _embeddings = []
        _index = None
        return

    _metadatas = [
        {"source": source, "index": i, "chars": len(chunk)}
        for i, chunk in enumerate(_chunks)
    ]

    embs = _encode(_chunks)
    _embeddings = embs.tolist()
    _index = faiss.IndexFlatIP(embs.shape[1])
    _index.add(embs)


def query_relevant_chunks(question: str, n_results: int = 3) -> List[str]:
    """Return top-N relevant chunk texts (plain strings)."""
    if _index is None or not _chunks:
        return []
    q = _encode([question])
    top_k = min(n_results, len(_chunks))
    _, indices = _index.search(q, top_k)
    return [_chunks[i] for i in indices[0] if i >= 0]


def query_relevant_documents(question: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Return top-N results as dicts with page_content + metadata (for Langchain Documents)."""
    if _index is None or not _chunks:
        return []
    q = _encode([question])
    top_k = min(n_results, len(_chunks))
    _, indices = _index.search(q, top_k)
    return [
        {"page_content": _chunks[i], "metadata": _metadatas[i]}
        for i in indices[0] if i >= 0
    ]


def get_all_chunks() -> List[str]:
    return _chunks


def get_all_metadatas() -> List[Dict[str, Any]]:
    return _metadatas


def get_all_embeddings():
    return _chunks, _embeddings
