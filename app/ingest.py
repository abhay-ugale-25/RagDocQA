import os
from typing import Optional

from dotenv import load_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

_EMBEDDINGS: Optional[GoogleGenerativeAIEmbeddings] = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a singleton instance of GoogleGenerativeAIEmbeddings."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        _EMBEDDINGS = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    return _EMBEDDINGS


def load_documents(file_path: str) -> list[Document]:
    """Sanitize path, validate existence and file extension, and load documents."""
    normalized_path = os.path.abspath(os.path.normpath(file_path))
    if not os.path.isfile(normalized_path):
        raise ValueError(f"File not found or is not a valid file: '{file_path}'")

    ext = os.path.splitext(normalized_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(normalized_path)
    elif ext == ".txt":
        loader = TextLoader(normalized_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .pdf or .txt")

    return loader.load()


def build_vectorstore(documents: list[Document]) -> tuple[Chroma, int]:
    """Split documents, validate non-empty chunks, and index into Chroma vector store using singleton embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    if len(chunks) == 0:
        raise ValueError("No extractable text found in document.")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vectorstore",
        collection_name="docqa",
    )

    return vectorstore, len(chunks)
