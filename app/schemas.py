# app/schemas.py — Pydantic request/response models for the API.

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response returned by the health-check endpoint."""

    status: str = Field(description="Current health status of the service, e.g. 'ok'.")


class IngestResponse(BaseModel):
    """Response returned after a document has been ingested."""

    status: str = Field(description="Outcome of the ingestion operation, e.g. 'success' or 'error'."
    )
    file: str = Field(description="Name of the ingested file."
    )
    chunks: int = Field(description="Number of text chunks produced and stored in the vector store."
    )


class QueryRequest(BaseModel):
    """Incoming request containing the user's natural-language question."""

    question: str = Field(description="The natural-language question to answer using retrieved documents."
    )


class QueryResponse(BaseModel):
    """Response returned after the retrieval-augmented generation pipeline runs."""

    answer: str = Field(description="The LLM-generated answer derived from retrieved document chunks."
    )
    sources: list[str] = Field(description="List of source document identifiers used to produce the answer."
    )
    reranked: Optional[bool] = Field(default=None,
        description="Whether the retrieved chunks were reranked before prompting the LLM. "
        "None indicates reranking status is unknown or not applicable."
    )
