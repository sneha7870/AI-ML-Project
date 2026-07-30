# RAG Chatbot — Capstone Project

A Retrieval-Augmented Generation chatbot that answers questions grounded in your own documents, deployable end-to-end on Render.

## Architecture

```
 User question
      │
      ▼
 ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
 │  Embed query │ --> │  FAISS similarity │ --> │ Top-K relevant   │
 │ (MiniLM-L6)  │     │  search over docs  │     │ chunks retrieved │
 └─────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                          ┌──────────────────────────┐
                                          │ Prompt = context + query  │
                                          │ sent to Claude / GPT      │
                                          └──────────────────────────┘
                                                          │
                                                          ▼
                                              Grounded answer + sources
```

**Offline (ingestion) pipeline** — run once, or whenever documents change:
`data/*.txt` → chunk (500 chars, 100 overlap) → embed each chunk locally with `sentence-transformers` (`all-MiniLM-L6-v2`) → store vectors in a FAISS index on disk.

**Online (query) pipeline** — runs per user message:
question → embed → FAISS search → top 4 chunks → build a prompt with that context → call an LLM (Claude by default, OpenAI as an alternative) → return the answer plus which source documents it drew from.

## Why this design
- **Embeddings run locally** (free, fast, no API cost) — only the final generation step calls a paid LLM, once per question. Keeps the whole capstone cheap to run and demo.
- **Provider-agnostic generation** — `rag_engine.py` supports both Anthropic Claude and OpenAI via `LLM_PROVIDER` env var, so you're not locked into one API/budget.
- **Grounded answers, not hallucinations** — the system prompt explicitly tells the LLM to answer only from retrieved context and say so if the answer isn't there, and the app returns which source files were used, so answers are auditable.

## Project structure
```
rag-chatbot-capstone/
├── ingest.py           # builds the vector index from data/
├── rag_engine.py        # core RAG logic (chunk, embed, retrieve, generate)
├── app.py                # Flask backend + chat API
├── templates/chat.html
├── static/style.css, chat.js
├── data/sample_docs/     # sample document so the pipeline works out of the box
├── requirements.txt
├── Dockerfile             # builds the index at image build time
├── render.yaml
├── .env.example
└── .gitignore
```

## 1. Run locally
```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in your ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# Replace data/sample_docs with your own knowledge base (.txt/.md files),
# or leave the sample doc in to test the pipeline first.
python ingest.py               # builds vector_store.faiss + chunks.pkl

python app.py                  # runs on http://localhost:5000
```
Open http://localhost:5000 and start chatting. Or test the API directly:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does the retrieval step work?"}'
```

**Note on this build**: I built and verified the document-loading and chunking logic directly (confirmed it correctly splits documents with overlap). The embedding model download (`sentence-transformers`, which pulls in PyTorch) is a large one-time download from Hugging Face — do that first `pip install` locally where you have full internet access; I couldn't complete it in this sandboxed environment due to network restrictions, but the code itself follows the standard, well-tested `sentence-transformers` + FAISS pattern.

## 2. Deploy on Render
1. Push to GitHub (add your real documents to `data/` first, or keep the sample).
2. Dashboard → **New** → **Blueprint** → connect your repo. Render reads `render.yaml` automatically.
3. In the Render dashboard, go to your service → **Environment** and add your real `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` + set `LLM_PROVIDER=openai`) as a secret — these are marked `sync: false` in `render.yaml` so they're never committed to your repo.
4. Click **Apply**. The Dockerfile runs `python ingest.py` at build time, so your knowledge base is baked into the deployed image — no separate ingestion step needed in production.
5. Note: I set the Render plan to `starter` rather than `free` in `render.yaml` — the embedding model needs more memory than Render's free 512MB tier reliably provides. Downgrade to `free` and test if you want to save cost, but expect possible OOM issues under load.

## 3. Rebuilding the index after changing documents
Since the Docker build step runs ingestion, just push a new commit with updated files in `data/` and Render will rebuild + redeploy automatically (`autoDeploy: true`). Locally, just rerun `python ingest.py`.

## For your capstone report
Good things to cover:
- **Chunking strategy tradeoffs**: why 500 chars with 100 overlap, and what happens with too-small (loses context) vs too-large (dilutes relevance, worse retrieval precision) chunks.
- **Why cosine similarity via normalized inner product** (what `faiss.normalize_L2` + `IndexFlatIP` does) rather than raw L2 distance — cosine similarity is scale-invariant, which matters for text embeddings.
- **Retrieval vs. generation failure modes**: distinguish "the answer wasn't in the retrieved chunks" (a retrieval problem — consider more chunks, better chunking, or hybrid keyword+vector search) from "the right chunks were retrieved but the LLM answered poorly" (a prompting/generation problem).
- **Evaluation**: consider building a small test set of Q&A pairs from your documents and manually scoring whether retrieved chunks actually contain the answer (retrieval recall) — a common, straightforward RAG evaluation approach for a capstone.
- **Extensions worth mentioning even if not implemented**: hybrid search (BM25 + vector), re-ranking retrieved chunks with a cross-encoder, conversation memory across turns (this demo is single-turn — each question is independent), streaming responses.
