from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response returned by the health-check endpoint."""

    status: str = Field(description="Current health status of the service, e.g. 'ok'.")


class IngestResponse(BaseModel):
    """Response returned after a document has been ingested."""

    status: str = Field(
        description="Outcome of the ingestion operation, e.g. 'success' or 'error'."
    )
    file: str = Field(description="Name of the ingested file.")
    chunks: int = Field(
        description="Number of text chunks produced and stored in the vector store."
    )


class QueryRequest(BaseModel):
    """Incoming request containing the user's natural-language question."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The natural-language question to answer using retrieved documents.",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of top reranked document chunks to retrieve and pass to the LLM.",
    )

    @field_validator("question", mode="before")
    @classmethod
    def strip_question_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v


class QueryResponse(BaseModel):
    """Response returned after the retrieval-augmented generation pipeline runs."""

    answer: str = Field(
        description="The LLM-generated answer derived from retrieved document chunks."
    )
    sources: list[str] = Field(
        description="List of source document identifiers used to produce the answer."
    )
    reranked: Optional[bool] = Field(
        default=None,
        description="Whether the retrieved chunks were reranked before prompting the LLM. "
        "None indicates reranking status is unknown or not applicable.",
    )
    latency_ms: float = Field(
        description="Execution latency of the query in milliseconds."
    )
    rerank_scores: list[float] = Field(
        default_factory=list,
        description="Cross-encoder relevance scores for the top retrieved documents.",
    )
