import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from app.ingest import get_embeddings

load_dotenv()
logger = logging.getLogger(__name__)

_RERANK_MODEL: Optional[CrossEncoder] = None


def get_rerank_model() -> CrossEncoder:
    """Return a singleton instance of the CrossEncoder reranking model."""
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        logger.info("Loading CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANK_MODEL


def __getattr__(name: str) -> Any:
    """Provide lazy backwards compatibility for module-level RERANK_MODEL access."""
    if name == "RERANK_MODEL":
        return get_rerank_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


PROMPT_TEMPLATE = """\
Given the following extracted parts of a long document and a question,
create a final answer with references ("SOURCES").
Answer ONLY using the provided context. If the answer is not contained
in the context, say "I don't know" — do NOT use outside knowledge.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["summaries", "question"],
)


class RerankedRetriever(BaseRetriever):
    base_retriever: Any
    cross_encoder: Any
    top_k: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def rerank(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        k = top_k if top_k is not None else self.top_k
        candidates = self.base_retriever.invoke(query)

        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        scored_docs = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )

        top_docs = []
        for score, doc in scored_docs[:k]:
            normalized_score = round(float(1.0 / (1.0 + np.exp(-float(score)))), 4)
            doc.metadata["rerank_score"] = normalized_score
            top_docs.append(doc)

        return top_docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.rerank(query, top_k=self.top_k)


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory="./vectorstore",
        collection_name="docqa",
        embedding_function=embeddings,
    )

    return vectorstore


def format_documents(docs: list[Document]) -> str:
    """Format retrieved documents into a context block for the LLM prompt."""
    if not docs:
        return "No relevant context found."

    formatted_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        formatted_parts.append(f"Content: {doc.page_content}\nSource: {source}")
    return "\n\n".join(formatted_parts)


def build_qa_chain(vectorstore: Chroma) -> RunnableLambda:
    retriever_vector = vectorstore.as_retriever(search_kwargs={"k": 8})

    all_docs = vectorstore.get()
    doc_texts = all_docs.get("documents", []) or []
    doc_metas = all_docs.get("metadatas", []) or [{}] * len(doc_texts)

    if doc_texts:
        bm25_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(doc_texts, doc_metas)
        ]
        retriever_bm25 = BM25Retriever.from_documents(bm25_docs, k=8)

        base_retriever = EnsembleRetriever(
            retrievers=[retriever_vector, retriever_bm25],
            weights=[0.6, 0.4],
        )
    else:
        base_retriever = retriever_vector

    retriever = RerankedRetriever(
        base_retriever=base_retriever,
        cross_encoder=get_rerank_model(),
        top_k=4,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    def execute_rag(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("question", "")
        top_k = inputs.get("top_k", retriever.top_k)

        # Retrieve documents and apply cross-encoder reranking
        docs = retriever.rerank(question, top_k=top_k)

        summaries = format_documents(docs)
        prompt_val = QA_PROMPT.format(question=question, summaries=summaries)
        response = llm.invoke(prompt_val)

        if isinstance(response.content, list):
            answer_text = "".join(
                item if isinstance(item, str) else item.get("text", "")
                for item in response.content
            )
        else:
            answer_text = str(response.content)

        return {
            "answer": answer_text.strip(),
            "source_documents": docs,
        }

    return RunnableLambda(execute_rag)
