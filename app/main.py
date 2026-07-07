# app/main.py — FastAPI app, routes, and lifespan management.

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from app.ingest import build_vectorstore, load_documents
from app.retriever import build_qa_chain, load_vectorstore
from app.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse

DATA_DIR = "./data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle of the application."""
    # ── Startup ──────────────────────────────────────────────
    print("Starting up RAG Document QA service...")

    # Load persisted vectorstore + qa_chain if data exists
    try:
        vectorstore = load_vectorstore()
        if vectorstore._collection.count() > 0:
            app.state.vectorstore = vectorstore
            app.state.qa_chain = build_qa_chain(vectorstore)
            print(f"Loaded vectorstore with {vectorstore._collection.count()} chunks.")
        else:
            app.state.vectorstore = None
            app.state.qa_chain = None
            print("Vectorstore is empty — ingest documents first.")
    except Exception:
        app.state.vectorstore = None
        app.state.qa_chain = None
        print("No persisted vectorstore found — ingest documents first.")
    yield
    # ── Shutdown ─────────────────────────────────────────────
    print("Shutting down RAG Document QA service...")
    # TODO (Phase 2): Persist / close the Chroma vectorstore here.


app = FastAPI(
    title="RAG Document QA",
    description="Retrieval-Augmented Generation pipeline for document question answering.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Health check ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Return a simple health-check response."""
    return HealthResponse(status="ok")


# ── Ingest ───────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile):
    """Upload a document, chunk it, embed it, and store in the vectorstore."""
    # Save uploaded file to data/
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Load and build vectorstore
    documents = load_documents(file_path)
    vectorstore, num_chunks = build_vectorstore(documents)

    # Store vectorstore and rebuild qa_chain
    app.state.vectorstore = vectorstore
    app.state.qa_chain = build_qa_chain(vectorstore)

    return IngestResponse(status="ingested", file=file.filename, chunks=num_chunks)


# ── Query ────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    """Answer a question using the retrieval-augmented generation pipeline."""
    if app.state.qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload a file via POST /ingest first.",
        )

    result = app.state.qa_chain.invoke({"question": body.question})

    # Extract unique source filenames from retrieved document metadata
    sources = list(
        {os.path.basename(doc.metadata.get("source", "")) for doc in result.get("source_documents", [])}
    )

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        reranked=True,
    )
