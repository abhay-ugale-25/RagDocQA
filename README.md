# 📄 RAG Document QA
![CI](https://github.com/abhay-ugale-25/RagDocQA/actions/workflows/ci.yml/badge.svg)


A **production-grade Retrieval-Augmented Generation** API that lets you upload documents (PDF/TXT), automatically chunks and embeds them into a vector store, and then answers natural-language questions grounded strictly in the uploaded content. The system combines semantic search with keyword matching via a hybrid retriever, reranks candidates using a cross-encoder model, and generates answers with Google's Gemini LLM — ensuring responses are always traceable back to source documents and never hallucinated from outside knowledge.

**Production features:** non-blocking async I/O via thread pool offloading, Pydantic v2 input validation with whitespace stripping, file path sanitization with size limits, singleton resource management, structured logging, configurable `top_k` retrieval, per-query latency tracking, cross-encoder rerank scores in API responses, and a deterministic LCEL pipeline replacing the deprecated `RetrievalQAWithSourcesChain`.

---

## Architecture

The system runs two pipelines through a single FastAPI server:

### Ingestion Pipeline — `POST /ingest`

```mermaid
flowchart LR
    A["📄 Upload PDF/TXT\n(≤ 20 MB)"] --> V["🛡️ Validate\n(sanitize path, check ext)"]
    V --> B["📖 Document Loader\n(PyPDF / TextLoader)"]
    B --> C["✂️ Text Splitter\n(800 chars, 150 overlap)"]
    C --> CHK["✅ Non-empty\nchunk check"]
    CHK --> D["🧠 Gemini Embeddings\n(singleton instance)"]
    D --> E[("💾 ChromaDB\n(persisted to disk)")]

    style A fill:#4a9eff,color:#fff,stroke:none
    style V fill:#e17055,color:#fff,stroke:none
    style B fill:#6c5ce7,color:#fff,stroke:none
    style C fill:#6c5ce7,color:#fff,stroke:none
    style CHK fill:#e17055,color:#fff,stroke:none
    style D fill:#00b894,color:#fff,stroke:none
    style E fill:#fdcb6e,color:#333,stroke:none
```

### Retrieval Pipeline — `POST /query`

```mermaid
flowchart TD
    Q["❓ User Question\n+ top_k (1–10)"] --> E1
    Q --> E2

    subgraph ensemble ["Ensemble Retriever"]
        E1["🔍 Vector Search\n(weight: 0.6, k=8)"]
        E2["🔤 BM25 Keyword Search\n(weight: 0.4, k=8)"]
    end

    E1 --> M["📋 Merged Candidates"]
    E2 --> M

    M --> R["⚖️ Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2)\nKeep top_k + scores"]
    R --> L["🤖 Gemini LLM via LCEL\n(context-only answering)"]
    L --> A["✅ Answer + Sources\n+ latency_ms + rerank_scores"]

    style Q fill:#4a9eff,color:#fff,stroke:none
    style ensemble fill:#f8f9fa,stroke:#ddd,stroke-width:2px
    style E1 fill:#6c5ce7,color:#fff,stroke:none
    style E2 fill:#6c5ce7,color:#fff,stroke:none
    style M fill:#636e72,color:#fff,stroke:none
    style R fill:#e17055,color:#fff,stroke:none
    style L fill:#00b894,color:#fff,stroke:none
    style A fill:#fdcb6e,color:#333,stroke:none
```

---

## Local Setup

### Prerequisites

- Python 3.10+
- A [Google AI API key](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone  https://github.com/abhay-ugale-25/RagDocQA.git
cd rag_dev

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure your API key
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
# Then edit .env and paste your GOOGLE_API_KEY
```

### Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

---

## API Reference

### `GET /health` — Health Check

```bash
curl http://127.0.0.1:8000/health
```

**Response** `200 OK`
```json
{
  "status": "ok"
}
```

---

### `POST /ingest` — Upload & Ingest a Document

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@./data/my_paper.pdf"
```

**Response** `200 OK`
```json
{
  "status": "ingested",
  "file": "my_paper.pdf",
  "chunks": 66
}
```

| Parameter | Type         | Description                                        |
|-----------|--------------|----------------------------------------------------|
| `file`    | `UploadFile` | PDF or TXT file to ingest (max 20 MB)              |

**Validation & Error Handling:**
- Filenames are sanitized via `os.path.basename()` to prevent path traversal.
- Files exceeding **20 MB** are rejected with `400 Bad Request`.
- Unsupported file extensions (anything other than `.pdf` / `.txt`) return `400`.
- Documents producing **zero text chunks** return `400` with `"No extractable text found in document."`

---

### `POST /query` — Ask a Question

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?", "top_k": 4}'
```

**Request Body**

| Field      | Type     | Required | Description                                                     |
|------------|----------|----------|-----------------------------------------------------------------|
| `question` | `string` | ✅       | Natural-language question (3–2000 chars, whitespace auto-stripped) |
| `top_k`    | `int`    | ❌       | Number of reranked chunks to use (1–10, default: 4)             |

**Response** `200 OK`
```json
{
  "answer": "The main contribution of this paper is the introduction of the Transformer...",
  "sources": ["my_paper.pdf"],
  "reranked": true,
  "latency_ms": 1523.47,
  "rerank_scores": [0.9987, 0.9412, 0.8876, 0.7103]
}
```

| Field           | Type           | Description                                              |
|-----------------|----------------|----------------------------------------------------------|
| `answer`        | `string`       | LLM-generated answer grounded in the uploaded documents  |
| `sources`       | `list[str]`    | Source filenames the answer was derived from              |
| `reranked`      | `bool\|null`   | Whether cross-encoder reranking was applied              |
| `latency_ms`    | `float`        | End-to-end query latency in milliseconds                 |
| `rerank_scores` | `list[float]`  | Cross-encoder relevance scores for each returned chunk   |

---

## Tech Stack

| Component             | Library                          | Why                                                                 |
|-----------------------|----------------------------------|---------------------------------------------------------------------|
| **Web Framework**     | FastAPI                          | Async-first, auto-generated OpenAPI docs, Pydantic v2 validation    |
| **ASGI Server**       | Uvicorn                          | Lightning-fast ASGI server, hot-reload for development              |
| **LLM**              | Gemini 3.1 Flash Lite            | Fast, low-latency responses; generous free-tier quota               |
| **Embeddings**        | Gemini Embedding 001             | High-quality dense embeddings optimized for retrieval               |
| **Vector Store**      | ChromaDB                         | Simple, file-persisted vector DB — zero infrastructure overhead     |
| **Orchestration**     | LangChain Core (LCEL)            | Deterministic `RunnableLambda` pipeline; no deprecated chain classes|
| **PDF Parsing**       | PyPDF                            | Reliable text extraction from PDF documents                         |
| **Keyword Search**    | rank-bm25                        | Classic TF-IDF style retriever for exact keyword matching            |
| **Reranker**          | sentence-transformers (CrossEncoder) | `ms-marco-MiniLM-L-6-v2` scores query-document relevance precisely |
| **Validation**        | Pydantic v2                      | Type-safe models with `Field` constraints, validators, and `ConfigDict` |
| **Async I/O**         | `run_in_threadpool`              | Offloads blocking calls to thread pool; keeps the event loop free   |
| **Logging**           | Python `logging`                 | Structured, leveled logging replacing `print()` statements          |
| **Config**            | python-dotenv                    | Loads API keys from `.env` — keeps secrets out of code              |

---

## How It Works

### The Problem

Imagine you have a 50-page research paper and you want to ask it questions like a conversation. You *could* paste the whole thing into ChatGPT — but what if it's too long? What if you have 100 documents? That's where RAG comes in.

### Step 1: Ingestion (Teaching the System Your Documents)

When you upload a PDF, the system doesn't just store it — it **reads and breaks it into small pieces** (called "chunks") of about 800 characters each, like cutting a book into index cards. Each card gets a small overlap with its neighbors so no idea gets split in half.

Then each chunk gets converted into a **list of numbers** (an "embedding") that captures its *meaning*. The sentence "The cat sat on the mat" and "A feline rested on the rug" would produce very similar number lists, even though they share almost no words. These embeddings are stored in a database (ChromaDB) for fast lookup later.

### Step 2: Retrieval (Finding the Right Index Cards)

When you ask a question, the system searches for relevant chunks using **two different strategies simultaneously**:

1. **Semantic search** (60% weight) — converts your question into the same kind of number list and finds chunks with similar *meaning*. Great for paraphrased concepts.

2. **Keyword search** (40% weight) — looks for chunks that share the same *exact words* as your question. Great for technical terms, names, or acronyms.

The results from both searches are merged together, giving you the best of both worlds.

### Step 3: Reranking (Picking the Best of the Best)

The merged list might have 10–15 candidate chunks, but not all are equally relevant. A specialized **reranker model** reads each chunk side-by-side with your question and assigns a relevance score. Only the **top `top_k` highest-scoring chunks** survive (configurable from 1–10, default 4) — this dramatically improves answer quality by filtering out noise. Each chunk's cross-encoder score is stored in its metadata and returned in the API response as `rerank_scores`.

### Step 4: Generation (Writing the Answer)

Those carefully-selected chunks are formatted into a context block and passed directly to the LLM via a deterministic **LCEL `RunnableLambda` pipeline** (replacing the deprecated `RetrievalQAWithSourcesChain`). The prompt instructs the LLM: *"Answer this question using ONLY the following context. If the answer isn't there, say you don't know."* The LLM (Gemini) then writes a natural-language answer and cites which source documents it used.

The result: an answer that's **always grounded in your actual documents**, never made up — with full latency metrics and reranking confidence scores.

---

## Project Structure

```
rag_dev/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app, routes, lifespan, async threadpool I/O
│   ├── ingest.py        ← validate → load → chunk → embed → store (singleton embeddings)
│   ├── retriever.py     ← ensemble search → rerank (with scores) → LCEL LLM pipeline
│   └── schemas.py       ← Pydantic v2 request/response models with validation
├── data/                ← uploaded documents land here
├── vectorstore/         ← ChromaDB persists embeddings here
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

