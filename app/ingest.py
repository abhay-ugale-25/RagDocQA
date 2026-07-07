# app/ingest.py — Load → chunk → embed → store pipeline for document ingestion.

import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()



def load_documents(file_path: str) -> list[Document]:
    """Load documents from a PDF or TXT file.

    Args:
        file_path: Path to the file to ingest (.pdf or .txt).

    Returns:
        A list of LangChain Document objects extracted from the file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .pdf or .txt")

    return loader.load()


def build_vectorstore(documents: list[Document]) -> tuple[Chroma, int]:
    """Split, embed, and store documents in a Chroma vectorstore.

    Args:
        documents: List of LangChain Document objects to process.

    Returns:
        A tuple of (Chroma vectorstore instance, number of chunks created).
    """
    # ── Split ────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # ── Embed + Store ────────────────────────────────────────
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vectorstore",
        collection_name="docqa",
    )

    return vectorstore, len(chunks)
