"""Document ingestion for the local RAG pipeline.

Pipeline: markdown/text files -> token-aware recursive chunks -> OpenAI
embeddings -> persistent ChromaDB collection.

``get_vectorstore()`` lives here (and only here) so the query chain reuses the
exact same collection, embedding function and persistence configuration.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import chromadb
import tiktoken
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    get_embeddings,
    load_settings,
)

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})


@lru_cache(maxsize=1)
def _get_token_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    """Measure text length in tokens, matching how the LLM budget is spent."""
    return len(_get_token_encoder().encode(text))


def build_text_splitter(
    chunk_size: int = CHUNK_SIZE + CHUNK_OVERLAP,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Build the shared token-aware recursive splitter.

    Separators are tried in order (paragraphs, lines, words, characters) so
    natural boundaries are respected whenever possible.

    RecursiveCharacterTextSplitter counts the overlap inside ``chunk_size``, so
    the window is ``CHUNK_SIZE + CHUNK_OVERLAP`` (550 tokens). A hard 500-token
    cap left the average at ~381 tokens on this corpus because paragraph
    boundaries rarely fill the window; 550 raises utilization while keeping the
    context small and still centered on the required ~500-token chunks.
    """
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_length,
        add_start_index=True,
    )


def discover_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return every supported file under ``data_dir`` in a deterministic order."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. Create it or set RAG_DATA_DIR."
        )
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load all .md/.txt files as UTF-8 documents.

    ``TextLoader`` is used per file instead of ``DirectoryLoader`` so ordering
    is deterministic and every file is explicitly decoded as UTF-8 (the corpus
    contains Spanish accents).
    """
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No .md or .txt files found in {data_dir}. Add documents before ingesting."
        )

    documents: list[Document] = []
    for path in files:
        loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            document.metadata["source"] = path.name
            document.metadata["source_path"] = path.relative_to(data_dir).as_posix()
            documents.append(document)
    return documents


def split_documents(documents: Iterable[Document]) -> list[Document]:
    """Split documents into chunks with normalized, per-file metadata."""
    splitter = build_text_splitter()
    chunks: list[Document] = []

    for document in documents:
        source = str(document.metadata.get("source") or "unknown")
        for chunk_index, chunk in enumerate(splitter.split_documents([document])):
            metadata = dict(chunk.metadata)
            metadata["source"] = source
            metadata["chunk_index"] = chunk_index
            # add_start_index=True normally provides this; keep a stable key.
            metadata.setdefault("start_index", -1)
            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

    return chunks


def _chunk_id(chunk: Document) -> str:
    return f"{chunk.metadata['source']}:{chunk.metadata['chunk_index']}"


def _collection_vector_count(client: chromadb.ClientAPI, name: str) -> int:
    try:
        return client.get_collection(name).count()
    except NotFoundError:
        return 0


def _collection_exists(client: chromadb.ClientAPI, name: str) -> bool:
    try:
        client.get_collection(name)
    except NotFoundError:
        return False
    return True


def get_vectorstore() -> Chroma:
    """Open the persistent Chroma collection used for both indexing and querying.

    This is the single retriever/vectorstore construction point in the project.
    langchain-chroma 1.x persists writes automatically; there is no ``persist()``
    method anymore and none must be called.
    """
    settings = load_settings()
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.vectorstore_dir),
        embedding_function=get_embeddings(),
        collection_metadata={
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )


def ingest_documents(force_reindex: bool = False) -> dict:
    """Index the corpus into ChromaDB, skipping work when it is already indexed.

    Args:
        force_reindex: delete the existing collection and rebuild it.

    Returns:
        A structured summary with ``files``, ``chunks``, ``collection``,
        ``skipped`` and the resolved storage/embedding details.
    """
    settings_vectorstore = VECTORSTORE_DIR
    client = chromadb.PersistentClient(path=str(settings_vectorstore))
    existing_vectors = _collection_vector_count(client, COLLECTION_NAME)
    file_count = len(discover_files(DATA_DIR))

    if existing_vectors > 0 and not force_reindex:
        summary = {
            "files": file_count,
            "chunks": existing_vectors,
            "collection": COLLECTION_NAME,
            "skipped": True,
            "vectorstore_dir": str(settings_vectorstore),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        _print_summary(summary, existing_vectors)
        return summary

    documents = load_documents(DATA_DIR)
    settings = load_settings()

    if force_reindex and _collection_exists(client, COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' (--force).")

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError(f"Documents in {DATA_DIR} produced no chunks; check their content.")

    # Same factory the query chain uses: one collection, one embedding function.
    vectorstore = get_vectorstore()

    # Stable ids make re-ingestion idempotent instead of silently duplicating.
    ids = [_chunk_id(chunk) for chunk in chunks]
    vectorstore.add_documents(documents=chunks, ids=ids)

    # langchain-chroma persists automatically on write; do NOT call .persist().
    persisted = vectorstore._collection.count()
    summary = {
        "files": len(documents),
        "chunks": persisted,
        "collection": settings.collection_name,
        "skipped": False,
        "vectorstore_dir": str(settings.vectorstore_dir),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    _print_summary(summary, 0)
    return summary


def _print_summary(summary: dict, existing_vectors: int) -> None:
    if summary["skipped"]:
        print("Indexing skipped: the collection already contains vectors.")
        print(f"  Existing vectors: {existing_vectors}")
        print(f"  Collection:       {summary['collection']}")
        print(f"  Location:         {summary['vectorstore_dir']}")
        print("Use --force to delete and rebuild the index.")
        return

    print("Ingestion completed.")
    print(f"  Files indexed:    {summary['files']}")
    print(f"  Chunks indexed:   {summary['chunks']}")
    print(f"  Collection:       {summary['collection']}")
    print(f"  Vector store:     {summary['vectorstore_dir']}")
    print(f"  Embedding model:  {summary['embedding_model']}")
    print(f"  Chunking:         {summary['chunk_size']} tokens / {summary['chunk_overlap']} overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local documents into ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing collection and rebuild the index.",
    )
    args = parser.parse_args()

    try:
        ingest_documents(force_reindex=args.force)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
