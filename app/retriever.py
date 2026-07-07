# app/retriever.py — Embed query → search → rerank → prompt → LLM retrieval pipeline.

import os
from typing import Any

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

# ── Cross-encoder reranker model (loaded once) ───────────────
RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Custom prompt ────────────────────────────────────────────
# Instructs the LLM to answer strictly from the provided context.
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
    """Wraps a base retriever and reranks results with a CrossEncoder."""

    base_retriever: Any  # EnsembleRetriever
    cross_encoder: Any  # CrossEncoder instance
    top_k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Step 1: Get merged candidates from ensemble
        candidates = self.base_retriever.invoke(query)

        if not candidates:
            return []

        # Step 2: Score every (query, chunk) pair with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Step 3: Sort descending by relevance score
        scored_docs = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )

        # Step 4: Keep only top_k
        return [doc for _, doc in scored_docs[: self.top_k]]


def load_vectorstore() -> Chroma:
    """Load the persisted Chroma vectorstore from disk.

    Uses the same embedding model and collection name as ingest.py
    to ensure embeddings are compatible.

    Returns:
        A Chroma vectorstore instance loaded from ./vectorstore.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    vectorstore = Chroma(
        persist_directory="./vectorstore",
        collection_name="docqa",
        embedding_function=embeddings,
    )

    return vectorstore


def build_qa_chain(vectorstore: Chroma) -> RetrievalQAWithSourcesChain:
    """Build a RetrievalQA chain from a Chroma vectorstore.

    Args:
        vectorstore: A Chroma instance to use as the retrieval backend.

    Returns:
        A RetrievalQAWithSourcesChain ready to answer questions.
    """
    # Vector retriever (semantic similarity)
    retriever_vector = vectorstore.as_retriever(search_kwargs={"k": 8})

    # BM25 retriever (keyword matching) built from all stored documents
    all_docs = vectorstore.get()
    from langchain_core.documents import Document

    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    retriever_bm25 = BM25Retriever.from_documents(bm25_docs, k=8)

    # Hybrid ensemble: 60% vector + 40% BM25
    ensemble = EnsembleRetriever(
        retrievers=[retriever_vector, retriever_bm25],
        weights=[0.6, 0.4],
    )

    # Cross-encoder reranking: score all candidates, keep top 4
    retriever = RerankedRetriever(
        base_retriever=ensemble,
        cross_encoder=RERANK_MODEL,
        top_k=4,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    return chain
