"""Regression tests for the config split-brain bug.

Before the fix, ``build_text_splitter`` and the retrieval helpers read
module-level constants captured at import time while collection metadata read
``load_settings()``. Any environment override made the two diverge. These tests
assert that every runtime path resolves configuration from ``load_settings()``.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import ingest
import rag_chain
from config import load_settings


class _FakeCollection:
    def __init__(self, count: int = 3, metadata: dict | None = None) -> None:
        self._count = count
        self.metadata = metadata or {"embedding_model": "text-embedding-3-small"}

    def count(self) -> int:
        return self._count


class _FakeVectorstore:
    def __init__(self) -> None:
        self._collection = _FakeCollection()
        self.retriever_kwargs: dict | None = None

    def as_retriever(self, **kwargs):
        self.retriever_kwargs = kwargs
        return object()


def test_splitter_honors_environment_overrides(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "600")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "120")

    settings = load_settings()
    splitter = ingest.build_text_splitter()

    # Derive expectations from load_settings() itself, not from literals, so the
    # test fails the moment the splitter and the settings disagree.
    expected_size = settings.chunk_size + settings.chunk_overlap
    expected_overlap = settings.chunk_overlap

    assert (settings.chunk_size, settings.chunk_overlap) == (600, 120)
    assert splitter._chunk_size == expected_size
    assert splitter._chunk_overlap == expected_overlap
    assert (splitter._chunk_size, splitter._chunk_overlap) == (720, 120)


def test_explicit_arguments_override_environment(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "600")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "120")

    splitter = ingest.build_text_splitter(chunk_size=111, chunk_overlap=22)

    assert splitter._chunk_size == 111
    assert splitter._chunk_overlap == 22


def test_collection_metadata_matches_settings(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "700")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "text-embedding-3-large")

    settings = load_settings()
    metadata = ingest._collection_metadata(settings)

    assert metadata == {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    assert metadata["chunk_size"] == 700
    assert metadata["chunk_overlap"] == 100

    # The splitter must be built from the same source, so the two cannot diverge.
    splitter = ingest.build_text_splitter()
    assert splitter._chunk_size == metadata["chunk_size"] + metadata["chunk_overlap"]
    assert splitter._chunk_overlap == metadata["chunk_overlap"]


def test_retriever_uses_settings_top_k(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "5")
    fake = _FakeVectorstore()
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: fake)
    monkeypatch.setattr(rag_chain, "get_vectorstore", lambda: fake)

    rag_chain._get_retriever.cache_clear()
    rag_chain._get_retriever()

    assert fake.retriever_kwargs is not None
    assert fake.retriever_kwargs["search_kwargs"]["k"] == 5


def test_deduplicate_default_follows_settings(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "3")
    documents = [
        Document(
            page_content=f"unique chunk {index}",
            metadata={"source": f"doc{index}.md", "chunk_index": index},
        )
        for index in range(6)
    ]

    assert len(documents) == 6
    assert len(rag_chain._deduplicate_documents(documents)) == 3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_TOP_K", "9"),
        ("RAG_CHUNK_SIZE", "100"),
    ],
)
def test_load_settings_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        load_settings()
