"""
rag_engine.py
---------------
Core RAG (Retrieval-Augmented Generation) logic:
  1. Load & chunk documents
  2. Embed chunks (local, free — sentence-transformers, no API cost)
  3. Store/search embeddings in a FAISS vector index
  4. Retrieve top-k relevant chunks for a query
  5. Pass retrieved context + query to an LLM (Anthropic Claude or OpenAI —
     configurable) to generate a grounded answer

Design note: embeddings are local (sentence-transformers) so indexing is
free and fast. Only the final generation step calls a paid LLM API, and only
once per user question — this keeps the capstone cheap to run and demo.
"""

import os
import glob
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, 384-dim, good quality-for-size tradeoff

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap so context isn't cut mid-thought
TOP_K = 4               # number of chunks retrieved per query


# ---------------------------------------------------------------------------
# 1. DOCUMENT LOADING & CHUNKING
# ---------------------------------------------------------------------------
def load_documents(data_dir="data"):
    """Loads all .txt and .md files from data_dir (recursively)."""
    paths = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)

    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            documents.append({"source": path, "text": f.read()})
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window character-based chunker with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunks(documents):
    """Splits every document into overlapping chunks, tagged with source."""
    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"text": chunk, "source": doc["source"]})
    return all_chunks


# ---------------------------------------------------------------------------
# 2 & 3. EMBEDDING + VECTOR INDEX
# ---------------------------------------------------------------------------
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def build_index(data_dir="data"):
    """Full ingestion pipeline: load docs -> chunk -> embed -> save FAISS index."""
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(
            f"No .txt or .md files found in '{data_dir}'. Add your knowledge-base "
            "documents there (see data/sample_docs for the expected format)."
        )

    chunks = build_chunks(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index ({index.ntotal} vectors, dim={embeddings.shape[1]}) to {INDEX_PATH}")
    return index, chunks


def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "No index found. Run `python ingest.py` first to build the vector store."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# 4. RETRIEVAL
# ---------------------------------------------------------------------------
def retrieve(query, index, chunks, top_k=TOP_K):
    embedder = get_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# 5. GENERATION (LLM call, provider-agnostic)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context. If the answer isn't in the context, say you don't have enough
information rather than guessing. Always be concise and cite which source(s) you
used when possible."""


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return f"""Context:
{context}

Question: {query}

Answer the question using only the context above."""


def generate_answer(query, retrieved_chunks, provider=None):
    """
    Calls an LLM to generate a grounded answer. Provider is chosen via the
    LLM_PROVIDER env var ('anthropic' or 'openai'), defaulting to 'anthropic'.
    Requires the corresponding API key as an environment variable
    (ANTHROPIC_API_KEY or OPENAI_API_KEY).
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
    prompt = build_prompt(query, retrieved_chunks)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'anthropic' or 'openai'.")


def answer_question(query, top_k=TOP_K, provider=None):
    """Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""
    index, chunks = load_index()
    retrieved = retrieve(query, index, chunks, top_k=top_k)
    answer = generate_answer(query, retrieved, provider=provider)
    return {
        "answer": answer,
        "sources": list({c["source"] for c in retrieved}),
        "retrieved_chunks": retrieved,
    }


if __name__ == "__main__":
    # Quick manual test from the command line
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    result = answer_question(q)
    print(f"\nQ: {q}\n")
    print(f"A: {result['answer']}\n")
    print(f"Sources: {result['sources']}")
