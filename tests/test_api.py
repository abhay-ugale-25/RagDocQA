import os
import shutil
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Mock environment variables BEFORE importing main to avoid initialization errors
os.environ["GOOGLE_API_KEY"] = "dummy-key-for-tests"

from app import ingest
from app import retriever

# Mock the embeddings and CrossEncoder instantiation at the module level
with patch("app.ingest.GoogleGenerativeAIEmbeddings"), \
     patch("app.retriever.CrossEncoder"):
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ingest_unsupported_file_extension():
    # Setup
    test_filename = "malicious.exe"
    file_path = os.path.join(DATA_DIR, test_filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    try:
        response = client.post(
            "/ingest",
            files={"file": (test_filename, b"dummy content", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json().get("detail", "")
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)


def test_query_no_qa_chain():
    # Set qa_chain to None to simulate empty state
    app.state.qa_chain = None
    response = client.post(
        "/query",
        json={"question": "What is the meaning of life?", "top_k": 4},
    )
    assert response.status_code == 400
    assert "No documents ingested yet" in response.json()["detail"]


def test_query_validation_constraints():
    app.state.qa_chain = MagicMock()

    # Test short question (<3 chars)
    response_short = client.post(
        "/query",
        json={"question": "Hi", "top_k": 4},
    )
    assert response_short.status_code == 422
    assert "String should have at least 3 characters" in str(response_short.json())

    # Test top_k out of bounds (>10)
    response_top_k = client.post(
        "/query",
        json={"question": "What is the meaning of life?", "top_k": 11},
    )
    assert response_top_k.status_code == 422
    assert "Input should be less than or equal to 10" in str(response_top_k.json())


def test_query_happy_path():
    app.state.qa_chain = MagicMock()
    app.state.qa_chain.invoke.return_value = {
        "answer": "Mocked grounded answer.",
        "source_documents": [
            Document(
                page_content="chunk 1",
                metadata={"source": "data/test.pdf", "rerank_score": 0.9512}
            )
        ]
    }

    response = client.post(
        "/query",
        json={"question": "What is the meaning of life?", "top_k": 4},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "Mocked grounded answer."
    assert data["sources"] == ["test.pdf"]
    assert data["reranked"] is True
    assert data["rerank_scores"] == [0.9512]
    assert data["latency_ms"] > 0
