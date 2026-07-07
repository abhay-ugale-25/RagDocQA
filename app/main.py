import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from app.ingest import build_vectorstore, load_documents
from app.retriever import build_qa_chain, load_vectorstore
from app.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse

DATA_DIR = "./data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up RAG Document QA service...")

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
    print("Shutting down RAG Document QA service...")


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
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    documents = load_documents(file_path)
    vectorstore, num_chunks = build_vectorstore(documents)

    app.state.vectorstore = vectorstore
    app.state.qa_chain = build_qa_chain(vectorstore)

    return IngestResponse(status="ingested", file=file.filename, chunks=num_chunks)


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    if app.state.qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload a file via POST /ingest first.",
        )

    result = app.state.qa_chain.invoke({"question": body.question})

    sources = list(
        {os.path.basename(doc.metadata.get("source", "")) for doc in result.get("source_documents", [])}
    )

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        reranked=True,
    )
