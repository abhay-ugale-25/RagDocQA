import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.ingest import build_vectorstore, load_documents
from app.retriever import build_qa_chain, load_vectorstore
from app.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = "./data"
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG Document QA service...")

    try:
        vectorstore = await run_in_threadpool(load_vectorstore)
        doc_count = vectorstore._collection.count()
        if doc_count > 0:
            app.state.vectorstore = vectorstore
            app.state.qa_chain = await run_in_threadpool(build_qa_chain, vectorstore)
            logger.info("Loaded vectorstore with %d chunks.", doc_count)
        else:
            app.state.vectorstore = None
            app.state.qa_chain = None
            logger.info("Vectorstore is empty — ingest documents first.")
    except Exception as e:
        app.state.vectorstore = None
        app.state.qa_chain = None
        logger.warning(
            "No persisted vectorstore found or error loading (%s) — ingest documents first.",
            e,
        )

    yield

    logger.info("Shutting down RAG Document QA service...")


app = FastAPI(
    title="RAG Document QA",
    description="Retrieval-Augmented Generation pipeline for document question answering.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    safe_filename = os.path.basename(file.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, safe_filename)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds the 20MB limit (uploaded: {len(content)} bytes).",
        )

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        documents = await run_in_threadpool(load_documents, file_path)
        vectorstore, num_chunks = await run_in_threadpool(build_vectorstore, documents)
        qa_chain = await run_in_threadpool(build_qa_chain, vectorstore)
    except ValueError as e:
        logger.warning("Validation error during ingestion of '%s': %s", safe_filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error processing document '%s'", safe_filename)
        raise HTTPException(
            status_code=500, detail=f"Failed to process and index document: {str(e)}"
        )

    app.state.vectorstore = vectorstore
    app.state.qa_chain = qa_chain

    logger.info("Successfully ingested '%s' (%d chunks).", safe_filename, num_chunks)
    return IngestResponse(status="ingested", file=safe_filename, chunks=num_chunks)


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    if app.state.qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload a file via POST /ingest first.",
        )

    start_time = time.perf_counter()

    try:
        result = await run_in_threadpool(
            app.state.qa_chain.invoke,
            {"question": body.question, "top_k": body.top_k},
        )
    except Exception as e:
        logger.exception("Error executing QA chain for question: %s", body.question)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate answer: {str(e)}"
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    source_docs = result.get("source_documents", [])

    seen_sources = set()
    sources = []
    for doc in source_docs:
        source_name = doc.metadata.get("source", "")
        if source_name:
            base_name = os.path.basename(source_name)
            if base_name and base_name not in seen_sources:
                seen_sources.add(base_name)
                sources.append(base_name)

    rerank_scores = [
        float(doc.metadata["rerank_score"])
        for doc in source_docs
        if "rerank_score" in doc.metadata
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        reranked=True,
        latency_ms=latency_ms,
        rerank_scores=rerank_scores,
    )
